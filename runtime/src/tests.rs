use super::*;
use std::ffi::CString;
use std::fs;
use std::sync::atomic::AtomicUsize;

static NEXT_FIXTURE: AtomicUsize = AtomicUsize::new(0);
struct Fixture {
    directory: PathBuf,
    paths: Vec<PathBuf>,
}
impl Fixture {
    fn new(shards: usize) -> Self {
        let directory = std::env::temp_dir().join(format!(
            "pmi-rust-{}-{}",
            std::process::id(),
            NEXT_FIXTURE.fetch_add(1, Ordering::SeqCst)
        ));
        fs::create_dir(&directory).unwrap();
        let paths = (0..shards)
            .map(|shard| {
                let path = directory.join(format!("shard-{shard}.bin"));
                let bytes: Vec<u8> = (0..512)
                    .map(|n| (n as u8).wrapping_add(shard as u8 * 17))
                    .collect();
                fs::write(&path, bytes).unwrap();
                path
            })
            .collect();
        Self { directory, paths }
    }
    fn state(&self, capacity: u64) -> State {
        State::open(&self.paths, capacity).unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.directory).unwrap();
    }
}
fn key(offset: u64, length: u64) -> Key {
    Key {
        shard: 0,
        offset,
        length,
    }
}
fn code<T>(result: Result<T>) -> i32 {
    result.err().expect("expected failure").code
}

#[test]
fn cache_hits_share_one_aligned_allocation() {
    let fixture = Fixture::new(1);
    let mut state = fixture.state(64);
    let first = state.request(key(3, 32)).unwrap();
    let second = state.request(key(3, 32)).unwrap();
    assert_eq!(first.bytes.pointer, second.bytes.pointer);
    assert_eq!(first.bytes.pointer.as_ptr() as usize % ALIGNMENT, 0);
    assert_eq!(first.bytes.bytes(), &(3..35).collect::<Vec<u8>>());
    let stats = state.stats();
    assert_eq!(
        (stats.hits, stats.misses, stats.io_bytes, stats.bytes_reused),
        (1, 1, 32, 32)
    );
    assert_eq!(
        (stats.current_bytes, stats.peak_bytes, stats.active_leases),
        (64, 64, 2)
    );
    drop(first);
    drop(second);
    assert_eq!(state.stats().active_leases, 0);
    assert_eq!(state.stats().current_bytes, 64);
}

#[test]
fn lru_eviction_and_reload_match_original_bytes() {
    let fixture = Fixture::new(1);
    let mut state = fixture.state(128);
    drop(state.request(key(0, 32)).unwrap());
    drop(state.request(key(32, 32)).unwrap());
    drop(state.request(key(0, 32)).unwrap());
    drop(state.request(key(64, 32)).unwrap());
    assert!(state.entries.contains_key(&key(0, 32)));
    assert!(!state.entries.contains_key(&key(32, 32)));
    let reloaded = state.request(key(32, 32)).unwrap();
    assert_eq!(reloaded.bytes.bytes(), &(32..64).collect::<Vec<u8>>());
    let stats = state.stats();
    assert_eq!(
        (stats.hits, stats.misses, stats.evictions, stats.io_bytes),
        (1, 4, 2, 128)
    );
    assert_eq!((stats.current_bytes, stats.peak_bytes), (128, 128));
}

#[test]
fn exact_ranges_and_overlaps_do_not_alias_cache_keys() {
    let fixture = Fixture::new(1);
    let mut state = fixture.state(192);
    let small = state.request(key(8, 8)).unwrap();
    let large = state.request(key(8, 16)).unwrap();
    let overlap = state.request(key(12, 8)).unwrap();
    assert_ne!(small.bytes.pointer, large.bytes.pointer);
    assert_eq!(small.bytes.bytes(), &large.bytes.bytes()[..8]);
    assert_eq!(overlap.bytes.bytes(), &(12..20).collect::<Vec<u8>>());
    assert_eq!((state.stats().entries, state.stats().io_bytes), (3, 32));
}

#[test]
fn shard_identity_is_part_of_the_cache_key() {
    let fixture = Fixture::new(2);
    let mut state = fixture.state(128);
    let first = state.request(key(0, 8)).unwrap();
    let second = state
        .request(Key {
            shard: 1,
            offset: 0,
            length: 8,
        })
        .unwrap();
    assert_eq!(first.bytes.bytes(), &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(second.bytes.bytes(), &[17, 18, 19, 20, 21, 22, 23, 24]);
    assert_eq!((state.stats().entries, state.stats().misses), (2, 2));
}

#[test]
fn blocked_capacity_failure_is_atomic_with_mixed_leases_and_pins() {
    let fixture = Fixture::new(1);
    let mut state = fixture.state(192);
    let leased = state.request(key(0, 32)).unwrap();
    drop(state.request(key(32, 32)).unwrap());
    drop(state.request(key(64, 32)).unwrap());
    state.pin(key(64, 32)).unwrap();
    assert_eq!(code(state.request(key(96, 128))), CAPACITY);
    let stats = state.stats();
    assert_eq!(
        (
            stats.entries,
            stats.evictions,
            stats.current_bytes,
            stats.io_bytes
        ),
        (3, 0, 192, 96)
    );
    assert_eq!(
        (stats.pinned_entries, stats.active_leases, stats.failures),
        (1, 1, 1)
    );
    assert!(state.entries.contains_key(&key(32, 32)));
    drop(leased);
    state.release(key(64, 32)).unwrap();
    drop(state.request(key(96, 128)).unwrap());
    assert_eq!(
        (state.stats().evictions, state.stats().peak_bytes),
        (2, 192)
    );
}

#[test]
fn pins_are_counted_and_unbalanced_release_is_rejected() {
    let fixture = Fixture::new(1);
    let mut state = fixture.state(64);
    assert_eq!(code(state.pin(key(0, 32))), NOT_FOUND);
    drop(state.request(key(0, 32)).unwrap());
    state.pin(key(0, 32)).unwrap();
    state.pin(key(0, 32)).unwrap();
    state.release(key(0, 32)).unwrap();
    assert_eq!(code(state.request(key(32, 32))), CAPACITY);
    state.release(key(0, 32)).unwrap();
    assert_eq!(code(state.release(key(0, 32))), NOT_FOUND);
    drop(state.request(key(32, 32)).unwrap());
    assert_eq!(state.stats().evictions, 1);
}

#[test]
fn lease_survives_provider_and_last_lease_frees_allocation() {
    let fixture = Fixture::new(1);
    let mut state = fixture.state(128);
    let ledger = Arc::clone(&state.ledger);
    let lease = state.request(key(5, 16)).unwrap();
    drop(state.request(key(32, 16)).unwrap());
    drop(state);
    assert_eq!(lease.bytes.bytes(), &(5..21).collect::<Vec<u8>>());
    assert_eq!(ledger.allocated.load(Ordering::SeqCst), 64);
    assert_eq!(ledger.leases.load(Ordering::SeqCst), 1);
    drop(lease);
    assert_eq!(ledger.allocated.load(Ordering::SeqCst), 0);
    assert_eq!(ledger.leases.load(Ordering::SeqCst), 0);
}

#[test]
fn invalid_ranges_fail_before_allocation_or_io() {
    let fixture = Fixture::new(1);
    let mut state = fixture.state(64);
    for invalid in [
        key(0, 0),
        key(0, u64::MAX),
        key(u64::MAX, 2),
        key(510, 4),
        Key {
            shard: 1,
            offset: 0,
            length: 1,
        },
    ] {
        assert_eq!(code(state.request(invalid)), INVALID);
    }
    assert_eq!(code(state.request(key(0, 65))), CAPACITY);
    let stats = state.stats();
    assert_eq!(
        (
            stats.entries,
            stats.current_bytes,
            stats.peak_bytes,
            stats.io_bytes
        ),
        (0, 0, 0, 0)
    );
    assert_eq!((stats.misses, stats.failures), (0, 6));
}

#[test]
fn truncated_read_accounts_actual_io_and_installs_no_partial_page() {
    let fixture = Fixture::new(1);
    let mut state = fixture.state(64);
    File::options()
        .write(true)
        .open(&fixture.paths[0])
        .unwrap()
        .set_len(10)
        .unwrap();
    assert_eq!(code(state.request(key(0, 32))), IO);
    let stats = state.stats();
    assert_eq!(
        (
            stats.entries,
            stats.io_bytes,
            stats.current_bytes,
            stats.peak_bytes,
            stats.failures
        ),
        (0, 10, 0, 64, 1)
    );
    fs::write(&fixture.paths[0], vec![9u8; 64]).unwrap();
    let lease = state.request(key(0, 32)).unwrap();
    assert_eq!(lease.bytes.bytes(), &[9u8; 32]);
    assert_eq!(state.stats().io_bytes, 42);
}

#[test]
fn invalid_provider_configuration_is_rejected() {
    let fixture = Fixture::new(1);
    assert_eq!(code(State::open(&[], 64)), INVALID);
    assert_eq!(code(State::open(&fixture.paths, 0)), INVALID);
    assert_eq!(
        code(State::open(std::slice::from_ref(&fixture.directory), 64)),
        INVALID
    );
    assert_eq!(
        code(State::open(&[fixture.directory.join("missing")], 64)),
        IO
    );
}

#[test]
fn concurrent_requests_share_storage_and_account_consistently() {
    let fixture = Fixture::new(1);
    let runtime = Arc::new(PmiRuntime {
        state: Mutex::new(fixture.state(128)),
    });
    std::thread::scope(|scope| {
        for thread in 0..8 {
            let runtime = Arc::clone(&runtime);
            scope.spawn(move || {
                for n in 0..100 {
                    let offset = ((thread + n) % 2) * 32;
                    let lease = lock(&runtime).unwrap().request(key(offset, 32)).unwrap();
                    assert_eq!(lease.bytes.bytes()[0], offset as u8);
                }
            });
        }
    });
    let stats = lock(&runtime).unwrap().stats();
    assert_eq!(
        (stats.hits, stats.misses, stats.io_bytes, stats.evictions),
        (798, 2, 64, 0)
    );
    assert_eq!(
        (stats.current_bytes, stats.peak_bytes, stats.active_leases),
        (128, 128, 0)
    );
}

#[test]
fn c_abi_transfers_lease_lifetime_and_returns_bounded_errors() {
    let fixture = Fixture::new(1);
    let path = CString::new(fixture.paths[0].to_str().unwrap()).unwrap();
    let mut runtime = ptr::null_mut();
    let mut lease = ptr::null_mut();
    let mut error = [7 as c_char; 32];
    unsafe {
        assert_eq!(
            pmi_runtime_create(
                &path.as_ptr(),
                1,
                64,
                &mut runtime,
                error.as_mut_ptr(),
                error.len()
            ),
            0
        );
        assert_eq!(error[0], 0);
        assert_eq!(
            pmi_runtime_request(
                runtime,
                0,
                4,
                8,
                &mut lease,
                error.as_mut_ptr(),
                error.len()
            ),
            0
        );
        assert_eq!(pmi_page_size(lease), 8);
        let mut stats = PmiRuntimeStats::default();
        assert_eq!(
            pmi_runtime_get_stats(runtime, &mut stats, ptr::null_mut(), 0),
            0
        );
        assert_eq!((stats.current_bytes, stats.active_leases), (64, 1));
        let mut failed_output = lease;
        assert_eq!(
            pmi_runtime_request(
                runtime,
                0,
                u64::MAX,
                8,
                &mut failed_output,
                error.as_mut_ptr(),
                1
            ),
            INVALID
        );
        assert!(failed_output.is_null());
        assert_eq!(error[0], 0);
        assert_eq!(error[1], 7);
        pmi_runtime_destroy(runtime);
        assert_eq!(
            std::slice::from_raw_parts(pmi_page_data(lease), 8),
            &[4, 5, 6, 7, 8, 9, 10, 11]
        );
        pmi_page_destroy(lease);
        assert_eq!(
            pmi_runtime_get_stats(ptr::null(), &mut stats, ptr::null_mut(), 0),
            INVALID
        );
        pmi_runtime_destroy(ptr::null_mut());
        pmi_page_destroy(ptr::null_mut());
        assert!(pmi_page_data(ptr::null()).is_null());
        assert_eq!(pmi_page_size(ptr::null()), 0);
    }
}

#[test]
fn transient_misses_are_bounded_and_cached_hits_preserve_pins() {
    let fixture = Fixture::new(2);
    let mut state = fixture.state(128);
    let a = Key {
        shard: 0,
        offset: 0,
        length: 64,
    };
    let b = Key {
        shard: 1,
        offset: 0,
        length: 64,
    };
    let c = Key {
        shard: 0,
        offset: 64,
        length: 64,
    };
    let retained = state.request(a).unwrap();
    state.pin(a).unwrap();
    let hit = state.request_policy(a, false).unwrap();
    assert!(Arc::ptr_eq(&retained.bytes, &hit.bytes));
    assert_eq!(state.stats().pinned_entries, 1);
    let transient = state.request_policy(b, false).unwrap();
    assert_eq!(transient.bytes.bytes()[0], 17);
    assert_eq!(state.stats().current_bytes, 128);
    assert_eq!(state.stats().entries, 1);
    assert_eq!(state.request_policy(c, false).err().unwrap().code, CAPACITY);
    drop(transient);
    assert_eq!(state.stats().current_bytes, 64);
    let again = state.request_policy(b, false).unwrap();
    assert_eq!(state.stats().misses, 4); // Includes capacity failure.
    drop(again);
    drop(hit);
    drop(retained);
    state.release(a).unwrap();
    assert_eq!(state.stats().current_bytes, 64);
    let lease = state.request_policy(b, false).unwrap();
    drop(state);
    assert_eq!(lease.bytes.bytes()[0], 17); // Survives runtime destruction.
}

#[test]
fn transient_short_read_and_bounds_cleanup() {
    let fixture = Fixture::new(1);
    let mut state = fixture.state(128);
    let key = Key {
        shard: 0,
        offset: 0,
        length: 64,
    };
    fs::write(&fixture.paths[0], [0u8; 2]).unwrap();
    assert_eq!(state.request_policy(key, false).err().unwrap().code, IO);
    assert_eq!(state.stats().current_bytes, 0);
    assert_eq!(state.stats().entries, 0);
    let invalid = Key { shard: 3, ..key };
    assert_eq!(
        state.request_policy(invalid, false).err().unwrap().code,
        INVALID
    );
    assert_eq!(state.stats().failures, 2);
}

#[test]
fn direct_range_alignment_overflow_and_interrupted_tail_reads() {
    use std::io::{self, ErrorKind};
    let a = direct::Alignment {
        memory: 512,
        offset: 512,
        statx: true,
    };
    let r = direct::range(63, 100, a).unwrap();
    assert_eq!((r.offset, r.prefix, r.layout.size()), (0, 63, 512));
    assert!(direct::range(0, 0, a).is_err());
    assert!(direct::range(u64::MAX, 1, a).is_err());
    assert!(direct::range(
        0,
        1,
        direct::Alignment {
            memory: 3,
            offset: 512,
            statx: true
        }
    )
    .is_err());
    let ledger = Arc::new(Ledger::default());
    let mut b = AlignedBytes::new(r.layout.size(), r.layout, Arc::clone(&ledger)).unwrap();
    assert_eq!(b.pointer.as_ptr() as usize % 512, 0);
    let mut calls = 0;
    let mut actual = 0;
    direct::read_aligned(b.bytes_mut(), &r, 512, &mut actual, |buf, offset| {
        calls += 1;
        assert_eq!(offset, 0);
        if calls == 1 {
            return Err(io::Error::from(ErrorKind::Interrupted));
        }
        for (i, x) in buf.iter_mut().take(200).enumerate() {
            *x = i as u8;
        }
        Ok(200) // Short padding tail is valid: all logical bytes arrived.
    })
    .unwrap();
    assert_eq!((calls, actual), (2, 200));
    assert_eq!(
        b.bytes()[63..163],
        (63..163).map(|x| x as u8).collect::<Vec<_>>()
    );
    drop(b);
    assert_eq!(ledger.allocated.load(Ordering::SeqCst), 0);
}

#[test]
fn direct_continuation_preserves_stricter_memory_alignment() {
    let a = direct::Alignment {
        memory: 4096,
        offset: 512,
        statx: true,
    };
    let r = direct::range(0, 600, a).unwrap();
    let ledger = Arc::new(Ledger::default());
    let mut b = AlignedBytes::new(r.layout.size(), r.layout, ledger).unwrap();
    let mut actual = 0;
    let mut calls = 0;
    let error = direct::read_aligned(
        b.bytes_mut(),
        &r,
        a.memory.max(a.offset),
        &mut actual,
        |_, o| {
            calls += 1;
            assert_eq!(o, 0);
            Ok(512)
        },
    )
    .unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof);
    assert_eq!((calls, actual), (1, 512));
}

#[test]
fn direct_short_continuations_and_errors_do_not_issue_unaligned_io() {
    use std::io::{self, ErrorKind};
    let r = direct::range(
        0,
        600,
        direct::Alignment {
            memory: 512,
            offset: 512,
            statx: true,
        },
    )
    .unwrap();
    let mut b = vec![0; 1024];
    let mut actual = 0;
    let mut calls = 0;
    let e = direct::read_aligned(&mut b, &r, 512, &mut actual, |_, o| {
        calls += 1;
        assert_eq!(o, 0);
        Ok(513)
    })
    .unwrap_err();
    assert_eq!(
        (e.kind(), calls, actual),
        (ErrorKind::UnexpectedEof, 1, 513)
    );
    actual = 0;
    calls = 0;
    let e = direct::read_aligned(&mut b, &r, 512, &mut actual, |_, o| {
        calls += 1;
        if calls == 1 {
            assert_eq!(o, 0);
            Ok(512)
        } else {
            assert_eq!(o, 512);
            Err(io::Error::from(ErrorKind::Other))
        }
    })
    .unwrap_err();
    assert_eq!((e.kind(), calls, actual), (ErrorKind::Other, 2, 512));
    assert!(!direct::unsupported(&io::Error::from(
        ErrorKind::PermissionDenied
    )));
    assert!(!direct::unsupported(&io::Error::from(
        ErrorKind::UnexpectedEof
    )));
}

#[test]
fn preferred_portable_fallback_is_explicit_and_required_fails_closed() {
    let f = Fixture::new(1);
    let mut s = f.state(128);
    s.direct_unavailable = true;
    let lease = s
        .request_io(key(1, 63), false, IoPolicy::Preferred)
        .unwrap();
    assert_eq!(
        lease.bytes.bytes(),
        (1..64).map(|x| x as u8).collect::<Vec<_>>()
    );
    assert_eq!(
        (
            s.io_stats.fallback_count,
            s.io_stats.direct_completed,
            s.io_stats.buffered_read_bytes
        ),
        (1, 0, 63)
    );
    drop(lease);
    assert_eq!(s.stats().current_bytes, 0);
    assert_eq!(
        s.request_io(key(0, 64), false, IoPolicy::Required)
            .err()
            .unwrap()
            .code,
        IO
    );
    assert_eq!(s.io_stats.fallback_count, 1);
    assert_eq!(s.stats().current_bytes, 0);
}

fn native_direct_fixture_available(state: &State) -> bool {
    match direct::open(&state.shards[0].path, &state.shards[0].file) {
        Ok(_) => true,
        Err(e) if direct::unsupported(&e) => {
            eprintln!("Native direct fixture unavailable: {e}; required-rejection tested; repeat using supported TMPDIR");
            false
        }
        Err(e) => panic!("native direct fixture capability error: {e}"),
    }
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[test]
fn direct_real_unaligned_ranges_shards_and_eof_preserve_exact_bytes() {
    let f = Fixture::new(2);
    let mut s = f.state(16384);
    if !native_direct_fixture_available(&s) {
        assert_eq!(
            s.request_io(key(0, 64), false, IoPolicy::Required)
                .err()
                .unwrap()
                .code,
            IO
        );
        return;
    }
    for shard in 0..2 {
        for (offset, length) in [(1, 65), (63, 128), (255, 64), (480, 32)] {
            let k = Key {
                shard,
                offset,
                length,
            };
            let lease = s.request_io(k, false, IoPolicy::Required).unwrap();
            assert_eq!(lease.bytes.pointer.as_ptr() as usize % 64, 0);
            assert_eq!(
                lease.bytes.bytes(),
                (offset..offset + length)
                    .map(|x| (x as u8).wrapping_add(shard as u8 * 17))
                    .collect::<Vec<_>>()
            );
            drop(lease);
            assert_eq!(s.stats().current_bytes, 0);
        }
    }
    assert_eq!(
        (
            s.io_stats.direct_completed,
            s.io_stats.fallback_count,
            s.io_stats.buffered_read_bytes
        ),
        (8, 0, 0)
    );
    assert!(s.io_stats.direct_read_bytes > s.io_stats.direct_logical_bytes);
    assert!(s.stats().peak_bytes <= s.stats().capacity_bytes);
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[test]
fn direct_staging_budget_blocks_atomically_and_cached_pins_survive() {
    let f = Fixture::new(1);
    let mut s = f.state(128);
    if !native_direct_fixture_available(&s) {
        assert_eq!(
            s.request_io(key(0, 64), false, IoPolicy::Required)
                .err()
                .unwrap()
                .code,
            IO
        );
        return;
    }
    let retained = s.request(key(0, 64)).unwrap();
    drop(retained);
    s.pin(key(0, 64)).unwrap();
    let hit = s.request_io(key(0, 64), false, IoPolicy::Required).unwrap();
    assert_eq!(
        (s.io_stats.direct_requests, s.stats().pinned_entries),
        (0, 1)
    );
    drop(hit);
    assert_eq!(
        s.request_io(key(64, 64), false, IoPolicy::Required)
            .err()
            .unwrap()
            .code,
        CAPACITY
    );
    assert_eq!(
        (
            s.stats().entries,
            s.stats().current_bytes,
            s.stats().evictions
        ),
        (1, 64, 0)
    );
    assert_eq!(s.io_stats.direct_read_bytes, 0);
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[test]
fn direct_truncation_cleanup_counts_partial_io_and_never_falls_back() {
    let f = Fixture::new(1);
    let mut s = f.state(16384);
    if !native_direct_fixture_available(&s) {
        assert_eq!(
            s.request_io(key(0, 64), false, IoPolicy::Required)
                .err()
                .unwrap()
                .code,
            IO
        );
        return;
    }
    fs::write(&f.paths[0], [7u8; 32]).unwrap();
    assert_eq!(
        s.request_io(key(0, 64), false, IoPolicy::Preferred)
            .err()
            .unwrap()
            .code,
        IO
    );
    assert_eq!(
        (
            s.stats().entries,
            s.stats().current_bytes,
            s.io_stats.fallback_count
        ),
        (0, 0, 0)
    );
    assert_eq!(s.io_stats.direct_read_bytes, 32);
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[test]
fn direct_reopened_shard_identity_is_verified_not_silently_replaced() {
    let f = Fixture::new(1);
    let mut s = f.state(16384);
    fs::rename(&f.paths[0], f.directory.join("old.bin")).unwrap();
    fs::write(&f.paths[0], [1u8; 512]).unwrap();
    assert_eq!(
        s.request_io(key(0, 64), false, IoPolicy::Preferred)
            .err()
            .unwrap()
            .code,
        IO
    );
    assert_eq!(
        (
            s.stats().entries,
            s.stats().current_bytes,
            s.io_stats.fallback_count
        ),
        (0, 0, 0)
    );
}

#[test]
fn direct_ffi_rejects_invalid_policy_and_separate_stats_have_bounded_output() {
    let f = Fixture::new(1);
    let path = CString::new(f.paths[0].to_str().unwrap()).unwrap();
    let mut runtime = ptr::null_mut();
    let mut lease = ptr::null_mut();
    let mut stats = PmiIoStats::default();
    unsafe {
        assert_eq!(
            pmi_runtime_create(&path.as_ptr(), 1, 8192, &mut runtime, ptr::null_mut(), 0),
            0
        );
        assert_eq!(
            pmi_runtime_request_direct(runtime, 0, 0, 64, 2, 1, &mut lease, ptr::null_mut(), 0),
            INVALID
        );
        assert!(lease.is_null());
        assert_eq!(
            pmi_runtime_get_io_stats(runtime, &mut stats, ptr::null_mut(), 0),
            0
        );
        assert_eq!(stats.direct_requests, 0);
        assert_eq!(
            pmi_runtime_get_io_stats(runtime, ptr::null_mut(), ptr::null_mut(), 0),
            INVALID
        );
        pmi_runtime_destroy(runtime);
    }
}

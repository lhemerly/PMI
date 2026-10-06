//! Model-independent, bounded file paging. No model layouts or backend kernels
//! belong here. Shards must remain immutable for the lifetime of a runtime.
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::collections::HashMap;
use std::ffi::{c_char, CStr};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

mod direct;

const ALIGNMENT: usize = 64;
const INVALID: i32 = 1;
const IO: i32 = 2;
const CAPACITY: i32 = 3;
const NOT_FOUND: i32 = 4;
const INTERNAL: i32 = 5;
const OUT_OF_MEMORY: i32 = 6;

#[derive(Debug)]
struct Error {
    code: i32,
    message: String,
}
impl Error {
    fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
    fn io(error: std::io::Error) -> Self {
        Self::new(IO, error.to_string())
    }
}
type Result<T> = std::result::Result<T, Error>;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PmiRuntimeStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub failures: u64,
    pub io_bytes: u64,
    pub bytes_reused: u64,
    pub current_bytes: u64,
    pub peak_bytes: u64,
    pub capacity_bytes: u64,
    pub entries: u64,
    pub pinned_entries: u64,
    pub active_leases: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct PmiIoStats {
    pub direct_requests: u64,
    pub direct_completed: u64,
    pub fallback_count: u64,
    pub direct_read_bytes: u64,
    pub buffered_read_bytes: u64,
    pub direct_logical_bytes: u64,
    pub staging_peak_bytes: u64,
    pub alignment_statx_requests: u64,
    pub alignment_known_fs_requests: u64,
}
#[derive(Clone, Copy, PartialEq)]
enum IoPolicy {
    Buffered,
    Preferred,
    Required,
}

#[derive(Default)]
struct Ledger {
    allocated: AtomicU64,
    peak: AtomicU64,
    leases: AtomicU64,
}

struct AlignedBytes {
    pointer: NonNull<u8>,
    length: usize,
    layout: Layout,
    ledger: Arc<Ledger>,
}
// The allocation is exclusively mutable during loading, then shared immutably.
// No mutable pointer or slice is exposed to a caller or stored separately.
unsafe impl Send for AlignedBytes {}
unsafe impl Sync for AlignedBytes {}
impl AlignedBytes {
    fn layout(length: u64) -> Result<Layout> {
        let length = usize::try_from(length)
            .map_err(|_| Error::new(INVALID, "page length does not fit address space"))?;
        if length == 0 {
            return Err(Error::new(INVALID, "zero-length page"));
        }
        let rounded = length
            .checked_add(ALIGNMENT - 1)
            .map(|n| n & !(ALIGNMENT - 1))
            .ok_or_else(|| Error::new(INVALID, "page allocation size overflow"))?;
        Layout::from_size_align(rounded, ALIGNMENT)
            .map_err(|_| Error::new(INVALID, "page allocation exceeds address space"))
    }
    fn new(length: usize, layout: Layout, ledger: Arc<Ledger>) -> Result<Self> {
        // Layout is nonzero and validated before any allocation or eviction.
        let pointer = NonNull::new(unsafe { alloc_zeroed(layout) })
            .ok_or_else(|| Error::new(OUT_OF_MEMORY, "page allocation failed"))?;
        let current = ledger
            .allocated
            .fetch_add(layout.size() as u64, Ordering::SeqCst)
            + layout.size() as u64;
        ledger.peak.fetch_max(current, Ordering::SeqCst);
        Ok(Self {
            pointer,
            length,
            layout,
            ledger,
        })
    }
    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.length) }
    }
    fn bytes_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.length) }
    }
}
impl Drop for AlignedBytes {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.pointer.as_ptr(), self.layout);
        }
        self.ledger
            .allocated
            .fetch_sub(self.layout.size() as u64, Ordering::SeqCst);
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
struct Key {
    shard: u32,
    offset: u64,
    length: u64,
}
struct Shard {
    path: PathBuf,
    direct: Option<(File, direct::Alignment)>,
    file: File,
    length: u64,
}
struct Entry {
    bytes: Arc<AlignedBytes>,
    last_use: u128,
    pins: u64,
}
struct State {
    shards: Vec<Shard>,
    entries: HashMap<Key, Entry>,
    capacity: u64,
    tick: u128,
    counters: PmiRuntimeStats,
    io_stats: PmiIoStats,
    #[cfg(test)]
    direct_unavailable: bool,
    ledger: Arc<Ledger>,
}

pub struct PmiRuntime {
    state: Mutex<State>,
}
pub struct PmiPageLease {
    bytes: Arc<AlignedBytes>,
}
impl PmiPageLease {
    fn new(bytes: Arc<AlignedBytes>) -> Self {
        bytes.ledger.leases.fetch_add(1, Ordering::SeqCst);
        Self { bytes }
    }
}
impl Drop for PmiPageLease {
    fn drop(&mut self) {
        self.bytes.ledger.leases.fetch_sub(1, Ordering::SeqCst);
    }
}

impl State {
    fn open(paths: &[PathBuf], capacity: u64) -> Result<Self> {
        if paths.is_empty() || paths.len() > u32::MAX as usize || capacity == 0 {
            return Err(Error::new(
                INVALID,
                "nonempty shards and nonzero cache capacity required",
            ));
        }
        let mut shards = Vec::with_capacity(paths.len());
        for path in paths {
            let file = File::open(path).map_err(Error::io)?;
            let metadata = file.metadata().map_err(Error::io)?;
            if !metadata.is_file() {
                return Err(Error::new(INVALID, "shard is not a regular file"));
            }
            shards.push(Shard {
                path: std::fs::canonicalize(path).map_err(Error::io)?,
                direct: None,
                file,
                length: metadata.len(),
            });
        }
        Ok(Self {
            shards,
            entries: HashMap::new(),
            capacity,
            tick: 0,
            counters: PmiRuntimeStats::default(),
            io_stats: PmiIoStats::default(),
            #[cfg(test)]
            direct_unavailable: false,
            ledger: Arc::new(Ledger::default()),
        })
    }
    fn request(&mut self, key: Key) -> Result<PmiPageLease> {
        self.request_policy(key, true)
    }
    fn request_policy(&mut self, key: Key, retain: bool) -> Result<PmiPageLease> {
        self.request_io(key, retain, IoPolicy::Buffered)
    }
    fn request_io(&mut self, key: Key, retain: bool, policy: IoPolicy) -> Result<PmiPageLease> {
        let result = self.load(key, retain, policy);
        if result.is_err() {
            self.counters.failures = self.counters.failures.saturating_add(1);
        }
        result
    }
    fn load(&mut self, key: Key, retain: bool, policy: IoPolicy) -> Result<PmiPageLease> {
        let layout = AlignedBytes::layout(key.length)?;
        let shard = self
            .shards
            .get(key.shard as usize)
            .ok_or_else(|| Error::new(INVALID, "shard index out of bounds"))?;
        let end = key
            .offset
            .checked_add(key.length)
            .ok_or_else(|| Error::new(INVALID, "page range overflow"))?;
        if end > shard.length {
            return Err(Error::new(INVALID, "page range exceeds shard bounds"));
        }
        let allocation = layout.size() as u64;
        if allocation > self.capacity {
            return Err(Error::new(
                CAPACITY,
                "aligned page allocation exceeds cache capacity",
            ));
        }
        self.tick = self.tick.saturating_add(1);
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_use = self.tick;
            self.counters.hits = self.counters.hits.saturating_add(1);
            self.counters.bytes_reused = self.counters.bytes_reused.saturating_add(key.length);
            return Ok(PmiPageLease::new(Arc::clone(&entry.bytes)));
        }
        self.counters.misses = self.counters.misses.saturating_add(1);

        let mut direct_range = None;
        if policy != IoPolicy::Buffered {
            self.io_stats.direct_requests = self.io_stats.direct_requests.saturating_add(1);
            let shard = &mut self.shards[key.shard as usize];
            #[cfg(test)]
            let enabled = !self.direct_unavailable;
            #[cfg(not(test))]
            let enabled = true;
            let opened = if !enabled {
                Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    "test portable direct fallback",
                ))
            } else if shard.direct.is_none() {
                direct::open(&shard.path, &shard.file).map(|value| {
                    shard.direct = Some(value);
                })
            } else {
                Ok(())
            };
            match opened {
                Ok(()) => {
                    direct_range = Some(
                        direct::range(key.offset, key.length, shard.direct.as_ref().unwrap().1)
                            .map_err(Error::io)?,
                    )
                }
                Err(e) if policy == IoPolicy::Preferred && direct::unsupported(&e) => {
                    self.io_stats.fallback_count = self.io_stats.fallback_count.saturating_add(1);
                }
                Err(e) => return Err(Error::io(e)),
            }
        }
        let total_charge = allocation
            .checked_add(direct_range.as_ref().map_or(0, |r| r.layout.size() as u64))
            .ok_or_else(|| Error::new(INVALID, "payload plus direct staging overflow"))?;
        if total_charge > self.capacity {
            return Err(Error::new(
                CAPACITY,
                "payload plus direct staging exceeds host capacity",
            ));
        }

        // Plan the entire eviction first. A blocked request must never evict
        // a subset of useful pages and then discover pinned/leased blockers.
        let current = self.ledger.allocated.load(Ordering::SeqCst);
        let available = self
            .capacity
            .checked_sub(current)
            .ok_or_else(|| Error::new(INTERNAL, "cache accounting exceeds capacity"))?;
        let needed = total_charge.saturating_sub(available);
        let mut candidates: Vec<_> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.pins == 0 && Arc::strong_count(&entry.bytes) == 1)
            .map(|(key, entry)| (entry.last_use, *key, entry.bytes.layout.size() as u64))
            .collect();
        candidates.sort_unstable_by_key(|&(tick, key, _)| (tick, key));
        let mut reclaim = 0u64;
        let mut count = 0;
        while reclaim < needed && count < candidates.len() {
            reclaim += candidates[count].2;
            count += 1;
        }
        if reclaim < needed {
            return Err(Error::new(
                CAPACITY,
                "cache capacity held by active leases or pins",
            ));
        }
        if retain {
            self.entries
                .try_reserve(1)
                .map_err(|_| Error::new(OUT_OF_MEMORY, "cache index allocation failed"))?;
        }
        for (_, key, _) in candidates.into_iter().take(count) {
            self.entries.remove(&key);
            self.counters.evictions = self.counters.evictions.saturating_add(1);
        }

        let mut bytes = AlignedBytes::new(key.length as usize, layout, Arc::clone(&self.ledger))?;
        let mut completed_direct = false;
        if let Some(range) = direct_range {
            let mut staging =
                AlignedBytes::new(range.layout.size(), range.layout, Arc::clone(&self.ledger))?;
            self.io_stats.staging_peak_bytes = self
                .io_stats
                .staging_peak_bytes
                .max(range.layout.size() as u64);
            let (file, alignment) = self.shards[key.shard as usize].direct.as_ref().unwrap();
            if alignment.statx {
                self.io_stats.alignment_statx_requests =
                    self.io_stats.alignment_statx_requests.saturating_add(1);
            } else {
                self.io_stats.alignment_known_fs_requests =
                    self.io_stats.alignment_known_fs_requests.saturating_add(1);
            }
            let mut actual = 0;
            let result = direct::read(file, staging.bytes_mut(), &range, *alignment, &mut actual);
            self.io_stats.direct_read_bytes =
                self.io_stats.direct_read_bytes.saturating_add(actual);
            self.counters.io_bytes = self.counters.io_bytes.saturating_add(actual);
            match result {
                Ok(()) => {
                    bytes.bytes_mut().copy_from_slice(
                        &staging.bytes()[range.prefix..range.prefix + range.length],
                    );
                    self.io_stats.direct_completed =
                        self.io_stats.direct_completed.saturating_add(1);
                    self.io_stats.direct_logical_bytes = self
                        .io_stats
                        .direct_logical_bytes
                        .saturating_add(key.length);
                    completed_direct = true;
                }
                Err(e)
                    if policy == IoPolicy::Preferred && actual == 0 && direct::unsupported(&e) =>
                {
                    self.io_stats.fallback_count = self.io_stats.fallback_count.saturating_add(1);
                }
                Err(e) => return Err(Error::io(e)),
            }
            // Staging drops here before a buffered fallback or upload lease escapes.
        }
        if !completed_direct {
            let file = &mut self.shards[key.shard as usize].file;
            file.seek(SeekFrom::Start(key.offset)).map_err(Error::io)?;
            let mut read = 0;
            while read < bytes.length {
                match file.read(&mut bytes.bytes_mut()[read..]) {
                    Ok(0) => {
                        return Err(Error::new(
                            IO,
                            "short page read; shard may have been truncated",
                        ))
                    }
                    Ok(n) => {
                        read += n;
                        self.counters.io_bytes = self.counters.io_bytes.saturating_add(n as u64);
                        self.io_stats.buffered_read_bytes =
                            self.io_stats.buffered_read_bytes.saturating_add(n as u64);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(Error::io(e)),
                }
            }
        }
        let bytes = Arc::new(bytes);
        let lease = PmiPageLease::new(Arc::clone(&bytes));
        if retain {
            self.entries.insert(
                key,
                Entry {
                    bytes,
                    last_use: self.tick,
                    pins: 0,
                },
            );
        }
        Ok(lease)
    }
    fn pin(&mut self, key: Key) -> Result<()> {
        let entry = self
            .entries
            .get_mut(&key)
            .ok_or_else(|| Error::new(NOT_FOUND, "page is not cached"))?;
        entry.pins = entry
            .pins
            .checked_add(1)
            .ok_or_else(|| Error::new(INVALID, "page pin count overflow"))?;
        Ok(())
    }
    fn release(&mut self, key: Key) -> Result<()> {
        let entry = self
            .entries
            .get_mut(&key)
            .ok_or_else(|| Error::new(NOT_FOUND, "page is not cached"))?;
        entry.pins = entry
            .pins
            .checked_sub(1)
            .ok_or_else(|| Error::new(NOT_FOUND, "page has no pin to release"))?;
        Ok(())
    }
    fn stats(&self) -> PmiRuntimeStats {
        PmiRuntimeStats {
            current_bytes: self.ledger.allocated.load(Ordering::SeqCst),
            peak_bytes: self.ledger.peak.load(Ordering::SeqCst),
            capacity_bytes: self.capacity,
            entries: self.entries.len() as u64,
            pinned_entries: self.entries.values().filter(|entry| entry.pins > 0).count() as u64,
            active_leases: self.ledger.leases.load(Ordering::SeqCst),
            ..self.counters
        }
    }
}

unsafe fn write_error(output: *mut c_char, capacity: usize, message: &str) {
    if !output.is_null() && capacity > 0 {
        let n = message.len().min(capacity - 1);
        ptr::copy_nonoverlapping(message.as_ptr(), output.cast::<u8>(), n);
        *output.add(n) = 0;
    }
}
fn ffi(error: *mut c_char, capacity: usize, operation: impl FnOnce() -> Result<()>) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(operation));
    match result {
        Ok(Ok(())) => {
            unsafe {
                write_error(error, capacity, "");
            }
            0
        }
        Ok(Err(failure)) => {
            unsafe {
                write_error(error, capacity, &failure.message);
            }
            failure.code
        }
        Err(_) => {
            unsafe {
                write_error(error, capacity, "Rust runtime internal panic");
            }
            INTERNAL
        }
    }
}
unsafe fn runtime<'a>(pointer: *const PmiRuntime) -> Result<&'a PmiRuntime> {
    pointer
        .as_ref()
        .ok_or_else(|| Error::new(INVALID, "null runtime"))
}
fn lock(runtime: &PmiRuntime) -> Result<std::sync::MutexGuard<'_, State>> {
    runtime
        .state
        .lock()
        .map_err(|_| Error::new(INTERNAL, "runtime lock poisoned"))
}
unsafe fn path_from_c(pointer: *const c_char) -> Result<PathBuf> {
    if pointer.is_null() {
        return Err(Error::new(INVALID, "null shard path"));
    }
    let string = CStr::from_ptr(pointer);
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        Ok(Path::new(std::ffi::OsStr::from_bytes(string.to_bytes())).to_owned())
    }
    #[cfg(not(unix))]
    {
        Ok(Path::new(
            string
                .to_str()
                .map_err(|_| Error::new(INVALID, "invalid UTF-8 shard path"))?,
        )
        .to_owned())
    }
}

/// Open immutable shards and create a runtime.
///
/// # Safety
/// `paths` must contain `count` valid NUL-terminated strings. `output` must be
/// writable. Optional error storage must hold `error_capacity` bytes. All input
/// and output regions must be valid and disjoint for the duration of this call.
#[no_mangle]
pub unsafe extern "C" fn pmi_runtime_create(
    paths: *const *const c_char,
    count: usize,
    capacity: u64,
    output: *mut *mut PmiRuntime,
    error: *mut c_char,
    error_capacity: usize,
) -> i32 {
    ffi(error, error_capacity, || {
        if output.is_null() {
            return Err(Error::new(INVALID, "null runtime output"));
        }
        *output = ptr::null_mut();
        if paths.is_null() || count == 0 || count > u32::MAX as usize {
            return Err(Error::new(INVALID, "invalid shard path array"));
        }
        let mut owned = Vec::with_capacity(count);
        for &path in std::slice::from_raw_parts(paths, count) {
            owned.push(path_from_c(path)?);
        }
        *output = Box::into_raw(Box::new(PmiRuntime {
            state: Mutex::new(State::open(&owned, capacity)?),
        }));
        Ok(())
    })
}
/// Destroy a runtime without invalidating outstanding page leases.
///
/// # Safety
/// `pointer` must be null or a live runtime returned by this library. Destroy
/// exactly once, after all calls using this runtime have completed.
#[no_mangle]
pub unsafe extern "C" fn pmi_runtime_destroy(pointer: *mut PmiRuntime) {
    if !pointer.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(pointer))));
    }
}
/// Read a range or acquire a lease of its cached allocation.
///
/// # Safety
/// `pointer` must be null or a live runtime. `output` and optional error storage
/// must be valid, disjoint writable regions of their declared sizes. Keep the
/// runtime alive until the call completes; destroy a successful lease once.
#[no_mangle]
pub unsafe extern "C" fn pmi_runtime_request(
    pointer: *mut PmiRuntime,
    shard: u32,
    offset: u64,
    length: u64,
    output: *mut *mut PmiPageLease,
    error: *mut c_char,
    error_capacity: usize,
) -> i32 {
    ffi(error, error_capacity, || {
        if output.is_null() {
            return Err(Error::new(INVALID, "null lease output"));
        }
        *output = ptr::null_mut();
        let lease = lock(runtime(pointer)?)?.request(Key {
            shard,
            offset,
            length,
        })?;
        *output = Box::into_raw(Box::new(lease));
        Ok(())
    })
}
/// Acquire a cached lease or a bounded, nonretained payload on a miss.
///
/// # Safety
/// `pointer` must be null or a live runtime. `output` and optional error storage
/// must be valid, disjoint writable regions of their declared sizes. Keep the
/// runtime alive until the call completes; destroy a successful lease once.
#[no_mangle]
pub unsafe extern "C" fn pmi_runtime_request_transient(
    pointer: *mut PmiRuntime,
    shard: u32,
    offset: u64,
    length: u64,
    output: *mut *mut PmiPageLease,
    error: *mut c_char,
    error_capacity: usize,
) -> i32 {
    ffi(error, error_capacity, || {
        if output.is_null() {
            return Err(Error::new(INVALID, "null lease output"));
        }
        *output = ptr::null_mut();
        let lease = lock(runtime(pointer)?)?.request_policy(
            Key {
                shard,
                offset,
                length,
            },
            false,
        )?;
        *output = Box::into_raw(Box::new(lease));
        Ok(())
    })
}
/// Acquire a cached lease or read an opt-in direct payload on a miss.
/// `retain` and `required` must be 0 or 1. Preferred mode permits only an
/// unsupported-direct fallback before any bytes arrive; required mode fails.
///
/// # Safety
/// `pointer` must be null or a live runtime. `output` and optional error storage
/// must be valid, disjoint writable regions of their declared sizes. Keep the
/// runtime alive until the call completes; destroy a successful lease once.
#[no_mangle]
pub unsafe extern "C" fn pmi_runtime_request_direct(
    pointer: *mut PmiRuntime,
    shard: u32,
    offset: u64,
    length: u64,
    retain: u8,
    required: u8,
    output: *mut *mut PmiPageLease,
    error: *mut c_char,
    error_capacity: usize,
) -> i32 {
    ffi(error, error_capacity, || {
        if output.is_null() {
            return Err(Error::new(INVALID, "null lease output"));
        }
        *output = ptr::null_mut();
        if retain > 1 || required > 1 {
            return Err(Error::new(INVALID, "invalid direct request policy"));
        }
        let lease = lock(runtime(pointer)?)?.request_io(
            Key {
                shard,
                offset,
                length,
            },
            retain != 0,
            if required != 0 {
                IoPolicy::Required
            } else {
                IoPolicy::Preferred
            },
        )?;
        *output = Box::into_raw(Box::new(lease));
        Ok(())
    })
}
/// Separate telemetry ABI leaves the existing runtime-statistics layout intact.
/// # Safety
/// Runtime must stay live; output/error regions must be valid and disjoint.
#[no_mangle]
pub unsafe extern "C" fn pmi_runtime_get_io_stats(
    pointer: *const PmiRuntime,
    output: *mut PmiIoStats,
    error: *mut c_char,
    error_capacity: usize,
) -> i32 {
    ffi(error, error_capacity, || {
        if output.is_null() {
            return Err(Error::new(INVALID, "null I/O stats output"));
        }
        *output = lock(runtime(pointer as *mut PmiRuntime)?)?.io_stats;
        Ok(())
    })
}
/// Add one pin to an already cached exact range.
///
/// # Safety
/// `pointer` must be null or a runtime kept alive throughout the call. Optional
/// error storage must be a disjoint writable region of `error_capacity` bytes.
#[no_mangle]
pub unsafe extern "C" fn pmi_runtime_pin(
    pointer: *mut PmiRuntime,
    shard: u32,
    offset: u64,
    length: u64,
    error: *mut c_char,
    error_capacity: usize,
) -> i32 {
    ffi(error, error_capacity, || {
        lock(runtime(pointer)?)?.pin(Key {
            shard,
            offset,
            length,
        })
    })
}
/// Release one pin on an already cached exact range.
///
/// # Safety
/// `pointer` must be null or a runtime kept alive throughout the call. Optional
/// error storage must be a disjoint writable region of `error_capacity` bytes.
#[no_mangle]
pub unsafe extern "C" fn pmi_runtime_release(
    pointer: *mut PmiRuntime,
    shard: u32,
    offset: u64,
    length: u64,
    error: *mut c_char,
    error_capacity: usize,
) -> i32 {
    ffi(error, error_capacity, || {
        lock(runtime(pointer)?)?.release(Key {
            shard,
            offset,
            length,
        })
    })
}
/// Snapshot cache counters and current owned allocations.
///
/// # Safety
/// `pointer` must be null or a runtime kept alive throughout the call. `output`
/// and optional error storage must be valid, disjoint writable regions of
/// their declared sizes.
#[no_mangle]
pub unsafe extern "C" fn pmi_runtime_get_stats(
    pointer: *const PmiRuntime,
    output: *mut PmiRuntimeStats,
    error: *mut c_char,
    error_capacity: usize,
) -> i32 {
    ffi(error, error_capacity, || {
        if output.is_null() {
            return Err(Error::new(INVALID, "null stats output"));
        }
        *output = lock(runtime(pointer)?)?.stats();
        Ok(())
    })
}
/// Return the aligned immutable payload address, or null for a null lease.
///
/// # Safety
/// `pointer` must be null or a live lease. Keep that lease alive while using the
/// returned pointer and do not mutate the allocation through a cast.
#[no_mangle]
pub unsafe extern "C" fn pmi_page_data(pointer: *const PmiPageLease) -> *const u8 {
    pointer
        .as_ref()
        .map_or(ptr::null(), |lease| lease.bytes.bytes().as_ptr())
}
/// Return the payload length, or zero for a null lease.
///
/// # Safety
/// `pointer` must be null or a lease kept alive for the duration of this call.
#[no_mangle]
pub unsafe extern "C" fn pmi_page_size(pointer: *const PmiPageLease) -> u64 {
    pointer
        .as_ref()
        .map_or(0, |lease| lease.bytes.length as u64)
}
/// Destroy one lease and allow its allocation to be evicted or freed.
///
/// # Safety
/// `pointer` must be null or a live lease returned by this library. Destroy
/// exactly once, after concurrent calls and uses of its data have completed.
#[no_mangle]
pub unsafe extern "C" fn pmi_page_destroy(pointer: *mut PmiPageLease) {
    if !pointer.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(pointer))));
    }
}

#[cfg(test)]
mod tests;

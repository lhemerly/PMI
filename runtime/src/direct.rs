//! Per-file opt-in direct reads. No driver/kernel/global settings are changed.
use std::alloc::Layout;
use std::fs::File;
use std::io::{self, ErrorKind};
use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub(super) struct Alignment {
    pub memory: usize,
    pub offset: usize,
    pub statx: bool,
}
pub(super) struct Range {
    pub offset: u64,
    pub prefix: usize,
    pub length: usize,
    pub layout: Layout,
}
pub(super) fn unsupported(e: &io::Error) -> bool {
    e.kind() == ErrorKind::Unsupported
        || (cfg!(target_os = "linux") && matches!(e.raw_os_error(), Some(22 | 38 | 95)))
}
pub(super) fn range(offset: u64, length: u64, alignment: Alignment) -> io::Result<Range> {
    if !alignment.memory.is_power_of_two()
        || !alignment.offset.is_power_of_two()
        || alignment.memory > 1024 * 1024
        || alignment.offset > 1024 * 1024
    {
        return Err(io::Error::new(
            ErrorKind::Unsupported,
            "invalid direct alignment",
        ));
    }
    let length = usize::try_from(length)
        .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "direct length overflow"))?;
    if length == 0 {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "zero direct length",
        ));
    }
    let mask = alignment.offset as u64 - 1;
    let prefix = (offset & mask) as usize;
    let required = prefix
        .checked_add(length)
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidInput, "direct range overflow"))?;
    let span = required
        .checked_add(alignment.offset - 1)
        .map(|n| n & !(alignment.offset - 1))
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidInput, "direct span overflow"))?;
    if span == 0 || (offset & !mask).checked_add(span as u64).is_none() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "direct end overflow",
        ));
    }
    let layout = Layout::from_size_align(span, alignment.memory.max(64))
        .map_err(|_| io::Error::new(ErrorKind::InvalidInput, "direct layout overflow"))?
        .pad_to_align();
    if (offset & !mask).checked_add(layout.size() as u64).is_none() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "direct padded end overflow",
        ));
    }
    Ok(Range {
        offset: offset & !mask,
        prefix,
        length,
        layout,
    })
}
// Reader returns actual bytes including alignment overlap. A short EOF tail is
// valid only once the complete logical range has arrived. Never issue an
// unaligned continuation after a partial direct read. EINTR preserves offsets.
pub(super) fn read_aligned<F>(
    buffer: &mut [u8],
    range: &Range,
    continuation_align: usize,
    actual: &mut u64,
    mut reader: F,
) -> io::Result<()>
where
    F: FnMut(&mut [u8], u64) -> io::Result<usize>,
{
    if !continuation_align.is_power_of_two() {
        return Err(io::Error::new(
            ErrorKind::InvalidInput,
            "invalid read alignment",
        ));
    }
    let required = range.prefix + range.length;
    let mut done = 0;
    while done < required {
        match reader(&mut buffer[done..], range.offset + done as u64) {
            Err(e) if e.kind() == ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
            Ok(n) if n > buffer.len() - done => {
                return Err(io::Error::new(
                    ErrorKind::InvalidData,
                    "reader exceeded buffer",
                ))
            }
            Ok(n) => {
                *actual = actual.saturating_add(n as u64);
                done += n;
                if done >= required {
                    return Ok(());
                }
                if n == 0 || done % continuation_align != 0 {
                    return Err(io::Error::new(
                        ErrorKind::UnexpectedEof,
                        "short direct payload; shard may be truncated",
                    ));
                }
            }
        }
    }
    Ok(())
}

#[cfg(all(target_os = "linux", target_env = "gnu", target_arch = "x86_64"))]
pub(super) fn open(path: &Path, original: &File) -> io::Result<(File, Alignment)> {
    use std::fs::OpenOptions;
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
    // Linux UAPI statx is 256 bytes. The stable DIOALIGN members have offsets
    // 0x98/0x9c; remaining fields are opaque. GNU libc >=2.28 exports statx.
    #[repr(C, align(8))]
    struct Statx {
        mask: u32,
        block: u32,
        before: [u8; 144],
        memory: u32,
        offset: u32,
        after: [u8; 96],
    }
    #[link(name = "dl")]
    unsafe extern "C" {
        fn dlsym(
            handle: *mut std::ffi::c_void,
            symbol: *const std::ffi::c_char,
        ) -> *mut std::ffi::c_void;
    }
    type StatxFn = unsafe extern "C" fn(i32, *const std::ffi::c_char, i32, u32, *mut Statx) -> i32;
    // Resolve optional libc support without a statx link requirement on older
    // GNU libc. RTLD_DEFAULT is NULL on this supported Linux GNU target.
    let address = unsafe { dlsym(std::ptr::null_mut(), c"statx".as_ptr()) };
    if address.is_null() {
        return Err(io::Error::new(
            ErrorKind::Unsupported,
            "libc statx unavailable",
        ));
    }
    let statx: StatxFn = unsafe { std::mem::transmute(address) };
    const O_DIRECT: i32 = 0o40000;
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(O_DIRECT)
        .open(path)?;
    let old = original.metadata()?;
    let new = file.metadata()?;
    if !new.is_file() || old.dev() != new.dev() || old.ino() != new.ino() {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "direct shard identity changed",
        ));
    }
    let mut s = Statx {
        mask: 0,
        block: 0,
        before: [0; 144],
        memory: 0,
        offset: 0,
        after: [0; 96],
    };
    // SAFETY: live descriptor, NUL-only pathname for AT_EMPTY_PATH, exact Linux
    // statx output layout, writable aligned storage; no borrowed buffer escapes.
    if unsafe { statx(file.as_raw_fd(), c"".as_ptr(), 0x1000, 0x2000, &mut s) } != 0 {
        return Err(io::Error::last_os_error());
    }
    if s.mask & 0x2000 == 0 || s.memory == 0 || s.offset == 0 {
        // Linux Btrfs reports sectorsize in fstatfs.f_bsize even on kernels
        // without STATX_DIOALIGN. Limit this ABI/profile to Linux GNU x86_64.
        // Compressed/inline extents may still cause kernel buffered fallback.
        unsafe extern "C" {
            fn fstatfs(fd: i32, out: *mut std::ffi::c_void) -> i32;
        }
        let mut info = [0u64; 64]; // Aligned oversized writable native-statfs storage.
        if unsafe { fstatfs(file.as_raw_fd(), info.as_mut_ptr().cast()) } != 0 {
            return Err(io::Error::last_os_error());
        }
        let sector = info[1] as usize;
        if info[0] != 0x9123683e || !sector.is_power_of_two() || sector > 1024 * 1024 {
            return Err(io::Error::new(
                ErrorKind::Unsupported,
                "filesystem lacks supported direct alignment",
            ));
        }
        return Ok((
            file,
            Alignment {
                memory: sector.max(64),
                offset: sector,
                statx: false,
            },
        ));
    }
    let alignment = Alignment {
        memory: s.memory as usize,
        offset: s.offset as usize,
        statx: true,
    };
    if !alignment.memory.is_power_of_two()
        || !alignment.offset.is_power_of_two()
        || alignment.memory > 1024 * 1024
        || alignment.offset > 1024 * 1024
    {
        return Err(io::Error::new(
            ErrorKind::Unsupported,
            "filesystem direct alignment is unsupported",
        ));
    }
    Ok((file, alignment))
}
#[cfg(not(all(target_os = "linux", target_env = "gnu", target_arch = "x86_64")))]
pub(super) fn open(_path: &Path, _original: &File) -> io::Result<(File, Alignment)> {
    Err(io::Error::new(
        ErrorKind::Unsupported,
        "verified direct I/O unavailable on this platform",
    ))
}
pub(super) fn read(
    file: &File,
    buffer: &mut [u8],
    range: &Range,
    alignment: Alignment,
    actual: &mut u64,
) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        // A continuation must preserve both the file-offset and buffer-pointer
        // alignment. Both are powers of two, so their maximum is sufficient.
        read_aligned(
            buffer,
            range,
            alignment.offset.max(alignment.memory),
            actual,
            |b, o| file.read_at(b, o),
        )
    }
    #[cfg(not(unix))]
    {
        let _ = (file, buffer, range, alignment, actual);
        Err(io::Error::new(
            ErrorKind::Unsupported,
            "positional direct read unavailable",
        ))
    }
}

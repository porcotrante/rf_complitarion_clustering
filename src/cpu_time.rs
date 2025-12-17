//! Provides a function to get the total CPU time (user + system) consumed by the current process.

use std::time::Duration;

/// Gets the total CPU time (user + system) consumed by the current process.
///
/// Returns `Duration::ZERO` if the platform is not supported or an error occurs.
pub fn get_cpu_time() -> Duration {
    #[cfg(unix)]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        // SAFETY: `usage.as_mut_ptr()` provides a valid pointer to `libc::rusage`.
        // `libc::RUSAGE_SELF` is a valid argument. The function returns 0 on success.
        if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } == 0 {
            // SAFETY: `getrusage` succeeded, so `usage` is initialized.
            let usage = unsafe { usage.assume_init() };
            let user_secs = usage.ru_utime.tv_sec as u64;
            let user_micros = usage.ru_utime.tv_usec as u32;
            let sys_secs = usage.ru_stime.tv_sec as u64;
            let sys_micros = usage.ru_stime.tv_usec as u32;

            Duration::from_secs(user_secs + sys_secs)
                + Duration::from_micros(user_micros as u64 + sys_micros as u64)
        } else {
            // Error occurred, return zero duration
            eprintln!("[WARN] Failed to get CPU usage via getrusage.");
            Duration::ZERO
        }
    }
    #[cfg(windows)]
    {
        use winapi::um::processthreadsapi::{GetCurrentProcess, GetProcessTimes};
        use winapi::shared::minwindef::FILETIME;
        use std::mem;

        let mut creation_time = mem::MaybeUninit::<FILETIME>::uninit();
        let mut exit_time = mem::MaybeUninit::<FILETIME>::uninit();
        let mut kernel_time = mem::MaybeUninit::<FILETIME>::uninit();
        let mut user_time = mem::MaybeUninit::<FILETIME>::uninit();

        // SAFETY: `GetCurrentProcess` returns a pseudo-handle that is always valid
        // for the current process. The pointers passed to `GetProcessTimes` are valid.
        let result = unsafe {
            GetProcessTimes(
                GetCurrentProcess(),
                creation_time.as_mut_ptr(),
                exit_time.as_mut_ptr(),
                kernel_time.as_mut_ptr(),
                user_time.as_mut_ptr(),
            )
        };

        if result != 0 {
            // SAFETY: `GetProcessTimes` succeeded, so kernel_time and user_time are initialized.
            let kernel_time = unsafe { kernel_time.assume_init() };
            let user_time = unsafe { user_time.assume_init() };

            // FILETIME represents 100-nanosecond intervals.
            let kernel_intervals = ((kernel_time.dwHighDateTime as u64) << 32) | (kernel_time.dwLowDateTime as u64);
            let user_intervals = ((user_time.dwHighDateTime as u64) << 32) | (user_time.dwLowDateTime as u64);

            let total_intervals = kernel_intervals + user_intervals;
            Duration::from_nanos(total_intervals * 100)
        } else {
            // Error occurred, return zero duration
            eprintln!("[WARN] Failed to get CPU usage via GetProcessTimes.");
            Duration::ZERO
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        // Platform not supported
        eprintln!("[WARN] CPU time measurement not supported on this platform.");
        Duration::ZERO
    }
}

use std::ffi::{CString, CStr};
use std::os::raw::{c_char, c_void};

unsafe extern "C" {
    fn opencc_open(config: *const c_char) -> *mut c_void;

    fn opencc_convert_utf8(
        opencc: *mut c_void,
        input: *const c_char,
        length: usize,
    ) -> *mut c_char;

    fn opencc_convert_utf8_free(ptr: *mut c_char);

    fn opencc_close(opencc: *mut c_void) -> i32;
}

pub fn t2s(text: &str) -> String{
    unsafe {
        let config = CString::new("t2s").unwrap();
        let handle = opencc_open(config.as_ptr());
        if handle.is_null() {
            panic!("opencc_open failed");
        }

        let input = CString::new(text).unwrap();
        let out = opencc_convert_utf8(
            handle,
            input.as_ptr(),
            input.as_bytes().len(),
        );

        if out.is_null() {
            panic!("opencc_convert_utf8 failed");
        }

        let result = CStr::from_ptr(out).to_string_lossy().into_owned();

        opencc_convert_utf8_free(out);
        opencc_close(handle);
        result
    }
}

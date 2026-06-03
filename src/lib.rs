mod error;
pub use error::LlmError;

pub fn _check() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_error_display() {
        let e = LlmError::Unauthorized;
        assert_eq!(e.to_string(), "unauthorized");
        let e2 = LlmError::InvalidRequest("bad param".into());
        assert!(e2.to_string().contains("bad param"));
    }
}

use std::env;

const MIN_CACHE_MIB: usize = 16;
const MAX_CACHE_MIB: usize = 8192;

pub(super) fn bytes(variable: &str, default_mib: usize) -> Result<usize, String> {
    let mib = match env::var(variable) {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|_| format!("{variable} must be an integer"))?,
        Err(env::VarError::NotPresent) => default_mib,
        Err(env::VarError::NotUnicode(_)) => return Err(format!("{variable} must be text")),
    };
    if !(MIN_CACHE_MIB..=MAX_CACHE_MIB).contains(&mib) {
        return Err(format!(
            "{variable} must be between {MIN_CACHE_MIB} and {MAX_CACHE_MIB}"
        ));
    }
    mib.checked_mul(1024 * 1024)
        .ok_or_else(|| "cache budget overflows this platform".into())
}

pub(super) fn max_positions(
    model_max_positions: usize,
    bytes_per_position: usize,
    cache_budget: usize,
) -> Result<usize, String> {
    if bytes_per_position == 0 {
        return Err("cache size per position is invalid".into());
    }
    let possible = cache_budget / bytes_per_position;
    if possible == 0 || (possible < 16 && model_max_positions > possible) {
        return Err("cache budget is too small for this model".into());
    }
    let power_of_two = 1_usize << (usize::BITS - 1 - possible.leading_zeros());
    Ok(model_max_positions.min(power_of_two))
}

#[cfg(test)]
mod tests {
    use super::max_positions;

    #[test]
    fn effective_context_respects_growable_cache_capacity() {
        let per_position = 12 * 2 * (2048 * 4 + std::mem::size_of::<Vec<f32>>());
        assert_eq!(
            max_positions(8192, per_position, 128 * 1024 * 1024).unwrap(),
            512
        );
        assert_eq!(
            max_positions(8192, per_position, 2048 * 1024 * 1024).unwrap(),
            8192
        );
        assert!(max_positions(8192, per_position, 1024).is_err());
        assert_eq!(max_positions(15, 100, 16 * 1024 * 1024).unwrap(), 15);
    }
}

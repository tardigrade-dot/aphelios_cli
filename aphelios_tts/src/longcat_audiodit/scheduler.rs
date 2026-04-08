use crate::longcat_audiodit::model::GuidanceMethod;

#[derive(Debug, Clone)]
pub struct DiffusionScheduler {
    pub steps: usize,
    pub cfg_strength: f64,
    pub guidance_method: GuidanceMethod,
    pub sigma: f64,
}

impl DiffusionScheduler {
    pub fn new(
        steps: usize,
        cfg_strength: f64,
        guidance_method: GuidanceMethod,
        sigma: f64,
    ) -> Self {
        Self {
            steps,
            cfg_strength,
            guidance_method,
            sigma,
        }
    }

    pub fn timesteps(&self) -> Vec<f32> {
        if self.steps == 0 {
            return vec![0.0];
        }
        // For Euler integration with N steps, we need N+1 points if we use the interval [t_i, t_i+1]
        // Python equivalent: dt = 1.0/steps; for i in range(steps): t = i/steps; ...
        (0..=self.steps)
            .map(|idx| idx as f32 / self.steps as f32)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestep_schedule_ascending() {
        // steps=3 → 3 timestep values matching torch.linspace(0, 1, 3): [0.0, 0.5, 1.0]
        let scheduler = DiffusionScheduler::new(3, 4.0, GuidanceMethod::Cfg, 0.0);
        let expected = vec![0.0, 0.5, 1.0];
        let actual = scheduler.timesteps();
        for (a, b) in actual.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "expected {}, got {}", b, a);
        }
    }

    #[test]
    fn timestep_schedule_matches_python_linspace() {
        // steps=16 → 16 values: [0.0, 1/15, 2/15, ..., 1.0]
        let scheduler = DiffusionScheduler::new(16, 4.0, GuidanceMethod::Cfg, 0.0);
        let actual = scheduler.timesteps();
        assert_eq!(actual.len(), 16);
        assert!((actual[0] - 0.0).abs() < 1e-5);
        assert!((actual[15] - 1.0).abs() < 1e-5);
        // Verify ascending
        for i in 1..actual.len() {
            assert!(actual[i] > actual[i - 1]);
        }
    }
}

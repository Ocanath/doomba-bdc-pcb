import numpy as np

# Constants
SQRT_3 = 1.732050807568877
ONE_BY_SQRT_3 = 0.577350269189626
TWO_BY_SQRT_3 = 1.154700538379252

def svm(alpha: np.ndarray, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Space Vector Modulation (SVM) implementation for three-phase inverter.
    Also known as Inverse Clarke transformation.
    
    Args:
        alpha: Alpha component of the voltage vector (array)
        beta: Beta component of the voltage vector (array)
        period_cnt: PWM period count (scalar)
        
    Returns:
        tuple: (tA, tB, tC, sector) - PWM compare values for phases A, B, C and sector number
               All outputs are arrays of the same shape as input arrays
    """
    # Ensure inputs are numpy arrays
    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    
    # Initialize output arrays
    shape = alpha.shape
    tA = np.zeros(shape)
    tB = np.zeros(shape)
    tC = np.zeros(shape)
    sector = np.zeros(shape, dtype=int)
    pwm_half_period = 1.0
    alpha_sqrt3 = alpha * SQRT_3
    
    # Determine sectors using boolean masks
    # Quadrant 1 or 2
    mask_q12 = beta >= 0
    # Quadrant 1
    mask_q1 = np.logical_and(mask_q12, alpha >= 0)
    # Sector 1
    mask_s1 = np.logical_and(mask_q1, beta <= alpha_sqrt3)
    # Sector 2 (from Q1)
    mask_s2_q1 = np.logical_and(mask_q1, beta > alpha_sqrt3)
    # Quadrant 2
    mask_q2 = np.logical_and(mask_q12, alpha < 0)
    # Sector 3
    mask_s3 = np.logical_and(mask_q2, beta <= -alpha_sqrt3)
    # Sector 2 (from Q2)
    mask_s2_q2 = np.logical_and(mask_q2, beta > -alpha_sqrt3)
    
    # Quadrant 3 or 4
    mask_q34 = beta < 0
    # Quadrant 4
    mask_q4 = np.logical_and(mask_q34, alpha >= 0)
    # Sector 5 (from Q4)
    mask_s5_q4 = np.logical_and(mask_q4, beta < -alpha_sqrt3)
    # Sector 6
    mask_s6 = np.logical_and(mask_q4, beta >= -alpha_sqrt3)
    # Quadrant 3
    mask_q3 = np.logical_and(mask_q34, alpha < 0)
    # Sector 5 (from Q3)
    mask_s5_q3 = np.logical_and(mask_q3, beta < alpha_sqrt3)
    # Sector 4
    mask_s4 = np.logical_and(mask_q3, beta >= alpha_sqrt3)
    
    # Assign sectors
    sector[mask_s1] = 1
    sector[mask_s2_q1] = 2
    sector[mask_s2_q2] = 2
    sector[mask_s3] = 3
    sector[mask_s4] = 4
    sector[mask_s5_q3] = 5
    sector[mask_s5_q4] = 5
    sector[mask_s6] = 6
    
    # Calculate PWM values for each sector
    for s in range(1, 7):
        mask = sector == s
        
        if np.any(mask):
            if s == 1:
                beta_by_sqrt3 = ONE_BY_SQRT_3 * beta[mask]
                two_by_sqrt3_beta = TWO_BY_SQRT_3 * beta[mask]
                
                t1 = (alpha[mask] - beta_by_sqrt3) * pwm_half_period
                t2 = two_by_sqrt3_beta * pwm_half_period
                tA[mask] = (pwm_half_period - t1 - t2) / 2
                tB[mask] = tA[mask] + t1
                tC[mask] = tB[mask] + t2
                
            elif s == 2:
                beta_by_sqrt3 = ONE_BY_SQRT_3 * beta[mask]
                
                t2 = (alpha[mask] + beta_by_sqrt3) * pwm_half_period
                t3 = (-alpha[mask] + beta_by_sqrt3) * pwm_half_period
                tB[mask] = (pwm_half_period - t2 - t3) / 2
                tA[mask] = tB[mask] + t3
                tC[mask] = tA[mask] + t2
                
            elif s == 3:
                beta_by_sqrt3 = ONE_BY_SQRT_3 * beta[mask]
                two_by_sqrt3_beta = TWO_BY_SQRT_3 * beta[mask]
                
                t3 = two_by_sqrt3_beta * pwm_half_period
                t4 = (-alpha[mask] - beta_by_sqrt3) * pwm_half_period
                tB[mask] = (pwm_half_period - t3 - t4) / 2
                tC[mask] = tB[mask] + t3
                tA[mask] = tC[mask] + t4
                
            elif s == 4:
                beta_by_sqrt3 = ONE_BY_SQRT_3 * beta[mask]
                two_by_sqrt3_beta = TWO_BY_SQRT_3 * beta[mask]
                
                t4 = (-alpha[mask] + beta_by_sqrt3) * pwm_half_period
                t5 = (-two_by_sqrt3_beta) * pwm_half_period
                tC[mask] = (pwm_half_period - t4 - t5) / 2
                tB[mask] = tC[mask] + t5
                tA[mask] = tB[mask] + t4
                
            elif s == 5:
                beta_by_sqrt3 = ONE_BY_SQRT_3 * beta[mask]
                
                t5 = (-alpha[mask] - beta_by_sqrt3) * pwm_half_period
                t6 = (alpha[mask] - beta_by_sqrt3) * pwm_half_period
                tC[mask] = (pwm_half_period - t5 - t6) / 2
                tA[mask] = tC[mask] + t5
                tB[mask] = tA[mask] + t6
                
            elif s == 6:
                beta_by_sqrt3 = ONE_BY_SQRT_3 * beta[mask]
                two_by_sqrt3_beta = TWO_BY_SQRT_3 * beta[mask]
                
                t6 = (-two_by_sqrt3_beta) * pwm_half_period
                t1 = (alpha[mask] + beta_by_sqrt3) * pwm_half_period
                tA[mask] = (pwm_half_period - t6 - t1) / 2
                tC[mask] = tA[mask] + t1
                tB[mask] = tC[mask] + t6
    
    tA = tA - 0.5
    tB = tB - 0.5
    tC = tC - 0.5
    return tA, tB, tC, sector 
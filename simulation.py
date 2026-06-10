import numpy as np
import matplotlib.pyplot as plt

class BlackSwanEnvironment:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.inverted = False
        
    def step(self, state, action):
        """
        Action Mapping: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        """
        x, y = state
        if self.inverted:
            if action == 0: y = max(0, y - 1)      # UP behaves as DOWN
            elif action == 1: x = max(0, x - 1)    # RIGHT behaves as LEFT
            elif action == 2: y = min(self.grid_size - 1, y + 1)
            elif action == 3: x = min(self.grid_size - 1, x + 1)
        else:
            if action == 0: y = min(self.grid_size - 1, y + 1)
            elif action == 1: x = min(self.grid_size - 1, x + 1)
            elif action == 2: y = max(0, y - 1)
            elif action == 3: x = max(0, x - 1)
            
        return (x, y)

class RTIContradictionDetector:
    def __init__(self, alpha=0.1, beta=2.5):
        self.alpha = alpha
        self.beta = beta
        self.mu_s = None
        self.sigma_s2 = None

    def evaluate(self, predicted_prob):
        eps = 1e-15
        p = max(eps, min(1.0 - eps, predicted_prob))
        s_t = -np.log(p)
        
        if self.mu_s is None:
            self.mu_s = s_t
            self.sigma_s2 = 1e-5
            return False, s_t

        sigma_s = np.sqrt(self.sigma_s2)
        is_contradiction = s_t > (self.mu_s + self.beta * sigma_s)
        
        if not is_contradiction:
            self.mu_s = (1 - self.alpha) * self.mu_s + self.alpha * s_t
            self.sigma_s2 = (1 - self.alpha) * self.sigma_s2 + self.alpha * ((s_t - self.mu_s) ** 2)
            
        return is_contradiction, s_t

# --- Plot Generation Function ---
def generate_thesis_plots():
    # Empirical data tracked straight from your "Black Swan Experimental Table"
    steps = np.array([0, 1, 2, 3, 4, 5])
    standard_model_ll = np.array([-5.0067, -4.9084, -4.8103, -4.7111, -4.6124, -4.5140])
    rti_model_ll = np.array([-5.0067, -0.6931, -0.6931, -0.6931, -0.6931, -0.6931])

    plt.figure(figsize=(8.5, 5))
    
    # Custom styling parameters matching modern journal criteria
    plt.plot(steps, rti_model_ll, marker='o', linewidth=2.5, color='#2b6cb0', 
             label='RTI Model (Non-Parametric Structural Patch)')
    plt.plot(steps, standard_model_ll, marker='s', linewidth=2.5, linestyle='--', color='#e53e3e', 
             label='Standard Model (Continuous Gradient Descent)')

    # Visual marker for the exact boundary index where the rule changes
    plt.axvline(x=0, color='#718096', linestyle=':', alpha=0.8)
    plt.text(0.1, -2.8, 'Black Swan Distribution Shift Triggered (t = 0)', 
             color='#4a5568', fontsize=9.5, fontstyle='italic')

    # Formatting and labels
    plt.title('World-Model Information Recovery Velocity (WMIR)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Time Steps Post-Contradiction Event ($t$)', fontsize=11)
    plt.ylabel('Model Log-Likelihood $\\log P(D_{new} \\mid M_t)$', fontsize=11)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.xticks(steps)
    plt.ylim(-5.5, 0.0)
    plt.legend(loc='lower right', frameon=True, facecolor='#f7fafc', edgecolor='#e2e8f0')
    
    plt.tight_layout()
    plt.savefig('RTI_vs_Gradient_Recovery_Velocity.png', dpi=300)
    print("Plot image saved successfully as 'RTI_vs_Gradient_Recovery_Velocity.png'")

if __name__ == "__main__":
    # Test the basic environment loop
    env = BlackSwanEnvironment()
    detector = RTIContradictionDetector()
    
    # Run data plotting sequence
    generate_thesis_plots()

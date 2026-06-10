import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(1,10,70).reshape(70,1)

# Create data
def plot_linear_regression(noise_controller: list | float):
    if isinstance(noise_controller, float):
        noise_controller = [noise_controller]

    fig, axs = plt.subplots(len(noise_controller), 2, figsize=(18,5))
    
    for i, noise in enumerate(noise_controller):
        y = 2 + 0.5*x + noise * np.random.randn(70,1)


        axs[i,0].scatter(x, y, label='Observations')

        axs[i,0].set_xlabel('x')
        axs[i,0].set_ylabel('y')

        axs[i,0].set_xlim([0,11])
        axs[i,0].set_ylim([0,10])

        axs[i,0].legend()


        X = x
        Y = y
        I = np.ones((70,1))
        A = np.hstack((I, X))


        # Least Squares Solution
        AT = A.T

        ATA = AT @ A
        ATA_inv = np.linalg.inv(ATA)

        ATY = AT @ y

        aHat = ATA_inv @ ATY
        a0 = aHat[0,0]
        a1 = aHat[1,0]

        print("aHat = ")
        print(aHat)

        axs[i,1].scatter(x, y, label='Observations')
        axs[i,1].set_xlabel('x')
        axs[i,1].set_ylabel('y')
        axs[i,1].set_xlim([0,11])
        axs[i,1].set_ylim([0,10])

        X1 = np.arange(1,11)
        Y1 = a0 + a1*X1
        S1 = np.sum((Y - (a0 + a1*X))**2)

        axs[i,1].plot(X1, Y1, '-k', label='Model')
        axs[i,1].legend()
        axs[i,1].set_title(f'Noise: {noise}, SSE: {S1:.2f}')

    plt.tight_layout()
    plt.show()


plot_linear_regression([0.1, 0.5, 1.0])
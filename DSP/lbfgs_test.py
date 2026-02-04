import numpy as np
import matplotlib.pyplot as plt


def optimise_with_reg_gradient_descent(points,y_true,num_epochs=10000,lr=1e-6):
    a_intial=1
    b_intial=0
    c_intial=0

    a=a_intial
    b=b_intial
    c=c_intial

    loss_history=[]
    for epoch in range(num_epochs):
        loss = np.sum((y_true - a*np.square(points) - b*points - c)**2)
        loss_history.append(loss)
        print("a,b,c are",a,b,c)
       # print(lr * np.sum((a*np.square(points) + b*points + c - y_true)*np.square(points)))
        a -= lr * np.sum((a*np.square(points) + b*points + c - y_true)*np.square(points))
        b -= lr * np.sum((a*np.square(points) + b*points + c - y_true)*points)
        c -= lr * np.sum((a*np.square(points) + b*points + c - y_true))
    
    print("converged a,b,c are",a,b,c)
    plt.plot(loss_history[500:])
    plt.show()
    return a,b,c

def main():
    points = np.linspace(0,10,1000)
    y = 3.5*np.square(points) + 5*np.random.randn(1000) + 2*points + 50

    a,b,c=optimise_with_reg_gradient_descent(points,y)
    
    
    plt.scatter(points, y)
    plt.plot(points, a*np.square(points) + b*points + c,color='red',linewidth=2)
    plt.show()


if __name__=="__main__": main()
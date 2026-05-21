# Модель: Метод хорд та метод Якобі (5 семестр)
# Автор: Кузьменко Костянтин Олександрович, група АІ-233

from flask import Flask, request, jsonify
import numpy as np
import os

app = Flask(__name__)

def f(x):
    return x**4 - 18*x**2 + 6

def secant_method(x0, x1, eps=0.01, max_iter=100):
    for k in range(max_iter):
        f0, f1 = f(x0), f(x1)
        if abs(f1 - f0) < 1e-12:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if abs(x2 - x1) < eps:
            return float(x2), k + 1
        x0, x1 = x1, x2
    return float(x1), max_iter

def jacobi_method(A, b, eps=0.01, max_iter=500):
    n = len(b)
    x = np.zeros(n)
    D = np.diag(A)
    R = A - np.diagflat(D)
    for k in range(max_iter):
        x_new = (b - R @ x) / D
        if np.linalg.norm(x_new - x, np.inf) < eps:
            return x_new.tolist(), k + 1
        x = x_new
    return x.tolist(), max_iter

@app.route('/calculate', methods=['GET', 'POST'])
def calculate():
    # Визначаємо метод 
    method = request.args.get('method', 'secant')

    if method == 'secant':
        # Параметри для методу хорд
        x0 = float(request.args.get('x0', -5.0))
        x1 = float(request.args.get('x1', 0.0))
        root, iters = secant_method(x0, x1)
        
        return jsonify({
            "method": "Secant (Метод хорд)",
            "params": {"x0": x0, "x1": x1},
            "result": {"root": root, "iterations": iters}
        })

    elif method == 'jacobi':
        # Дані для методу Якобі 
        A = np.array([
            [3, 1, 0, 0],
            [1, 4, -1, 0],
            [0, -1, 5, 1],
            [0, 0, 1, 2]
        ], float)
        b = np.array([5, 3, 12, 6], float)
        
        solution, iters = jacobi_method(A, b)
        
        return jsonify({
            "method": "Jacobi (Метод Якобі)",
            "result": {
                "vector_x": solution,
                "iterations": iters
            }
        })

    else:
        return jsonify({"error": "Unknown method"}), 400

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

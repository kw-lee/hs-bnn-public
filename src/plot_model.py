import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb
import argparse
import joblib
import sys
import jax.numpy as jnp

sys.path.append(".")
from src.hs_bnn import HSBnn, predict

sb.set_context("paper", rc={"lines.linewidth": 5, "lines.markersize":10, 'axes.labelsize': 8,
   'text.fontsize': 8,
   'legend.fontsize': 15,
   'xtick.labelsize': 15,
   'ytick.labelsize': 15,
   'ylabel.fontsize':15,
   'xlabel.fontsize':15,
   'text.usetex': False,
    'axes.titlesize' : 25,
    'axes.labelsize' : 25,  })
sb.set_style("darkgrid")

def plot_model(mlp, posterior_mode=False, export_path=None):
    plt.figure(figsize = (6, 6))
    X = jnp.sort(mlp.X, axis=0)
    axx = plt.gca()

    y_preds = []
    for i in range(50):
        y_preds.append(predict(mlp, X, seed=i))
    y_pred = jnp.concatenate(y_preds, axis=1)
    
    pred_mean = y_pred.mean(axis=1).reshape(-1)
    pred_std = y_pred.std(axis=1).reshape(-1)
    plt.fill_between(X.reshape(-1), y1=pred_mean - 3 * pred_std, y2=pred_mean + 3 * pred_std,
        alpha=0.2)
    plt.plot(X, pred_mean, 'r--')
    plt.scatter(mlp.X, mlp.y, c='blue')
    if export_path: 
        plt.savefig(export_path)
    else:
        plt.show(block=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("r_path")
    args = parser.parse_args()
    r_path = args.r_path
    model = joblib.load(r_path)
    plot_model(model, export_path=f'{r_path}.pdf')

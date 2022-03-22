import matplotlib.pyplot as plt

from Classes.results import Results
from main import temporary_results_name

r = Results()
r.readResult(temporary_results_name)
r.plotResults()
plt.savefig(f"./out/{temporary_results_name}.png")

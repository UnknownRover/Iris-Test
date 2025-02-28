{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b2387ce5-05a1-4f36-adec-90b695bd3ff4",
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask, render_template\n",
    "from kmeans_iris import kmeans_clustering, \n",
    "plot_2d_scatter, plot_3d_scatter\n",
    "\n",
    "app = Flask(__name__)\n",
    "\n",
    "@app.route('/')\n",
    "def home():\n",
    "    df, target_names = kmeans_clustering()\n",
    "    plot_2d_scatter(df)\n",
    "    plot_3d_scatter(df)\n",
    "    return render_template(\"index.html\")\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(debug=True)\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

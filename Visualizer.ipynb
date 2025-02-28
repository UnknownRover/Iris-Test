{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "d7ff8c6e-70ec-44a6-b3e1-98a2123bb2a4",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.cluster import KMeans\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.datasets import load_iris\n",
    "import plotly.express as px\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "3ff2dbc0-0557-481e-be9b-de392f949d6f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def kmeans_clustering():\n",
    "    iris = load_iris()\n",
    "    df = pd.DataFrame(iris.data, columns=iris.feature_names)\n",
    "    scaler = StandardScaler()\n",
    "    df_scaled = scaler.fit_transform(df)\n",
    "    kmeans = KMeans(n_clusters = 3, random_state=42, n_init=10)\n",
    "    df['Cluster'] = kmeans.fit_predict(df_scaled)\n",
    "    return df, iris.target_names"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "4d5dec7d-0969-4f60-a990-4ca819b804e3",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_2d_scatter(df):\n",
    "    plt.figure(figsize=(8,6))\n",
    "    sns.scatterplot(x=df['sepal length (cm)'],\n",
    "        y=df['sepal width (cm)'], hue=df['Cluster'], palette = 'viridis')\n",
    "    plt.xlabel(\"Sepal Length (cm)\")\n",
    "    plt.ylabel(\"Sepal Width (cm)\")\n",
    "    plt.title(\"K-Means Clustering (2D View)\")\n",
    "    plt.savefig(\"static/plot_2d.png\")\n",
    "    plt.close()\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "5676cfe2-7a54-4aa1-89a2-bd0bff214caf",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_3d_scatter(df):\n",
    "    fig = px.scatter_3d(df, x='sepal length (cm)', y ='sepal width (cm)',\n",
    "                        z='petal length (cm)', color =df['Cluster'].astype(str),\n",
    "                        title=\"K-Means Clustering (3D View)\")\n",
    "    fig.write_html(\"static/plot_3d.html\")\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "5d2c8078-8658-45d0-9617-cf5fd9976922",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\anaconda3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:1429: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "df,sample = kmeans_clustering()\n",
    "plot_2d_scatter(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "585ad6af-cfce-44ab-abfe-d63e6fd4da30",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\ProgramData\\anaconda3\\Lib\\site-packages\\sklearn\\cluster\\_kmeans.py:1429: UserWarning:\n",
      "\n",
      "KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=1.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "df,sample = kmeans_clustering()\n",
    "plot_3d_scatter(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ad9db8c9-13a7-4d2d-9b9e-0908806874c0",
   "metadata": {},
   "outputs": [],
   "source": []
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

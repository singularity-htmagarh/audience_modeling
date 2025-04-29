from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from utils import generate_synthetic_customer_data

data = generate_synthetic_customer_data()
features = ['age','income','visits_per_month','avg_order_value']
X = StandardScaler().fit_transform(data[features])

kmeans = KMeans(n_clusters=4, random_state=42)
data['segment'] = kmeans.fit_predict(X)

# Visualize
plt.scatter(X[:,0],X[:,1],c=data['segment'],cmap='viridis')
plt.title("Customer Segments")
plt.xlabel("Age (scaled)")
plt.ylabel("Income (scaled)")
plt.show()

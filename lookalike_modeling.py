from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from utils import generate_synthetic_customer_data

data = generate_synthetic_customer_data()
features = ['age','income','visits_per_month','avg_order_value']
X = data[features]
y = data['is_high_value']

X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y,test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimator=100, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# Predict Lookalikes
data['lookalike_score'] = clf.predict_proba(X)[:,1]
lookalikes = data[data[]>0.7]
print(f"Identified {len(lookalikes)} lookalike customers.")


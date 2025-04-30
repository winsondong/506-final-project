from Final_project_RandomForest import train_for_visualizations, standard_amenities
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import learning_curve
from sklearn.metrics import mean_squared_error, r2_score

# Load raw data for initial visualizations
def load_data(file_path="airbnbListingsData.csv"):
    df = pd.read_csv(file_path)
    df["price"] = df["price"].replace(r'[\$,]', '', regex=True).astype(float)
    df = df[df["price"] < 1000]
    return df

# Initial visualizations
def plot_price_histogram(df):
    plt.figure(figsize=(10,6))
    plt.hist(df["price"], bins=50, color='skyblue', edgecolor='black')
    plt.title("Distribution of Airbnb Prices in NYC")
    plt.xlabel("Price ($)")
    plt.ylabel("Number of Listings")
    plt.grid(True)
    plt.show()

def plot_scatter_price_vs_rooms(df):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(df["bedrooms"], df["price"], alpha=0.5)
    plt.title("Price vs Bedrooms")
    plt.xlabel("Bedrooms")
    plt.ylabel("Price ($)")
    plt.subplot(1, 2, 2)
    plt.scatter(df["bathrooms"], df["price"], alpha=0.5, color='orange')
    plt.title("Price vs Bathrooms")
    plt.xlabel("Bathrooms")
    plt.ylabel("Price ($)")
    plt.tight_layout()
    plt.show()

def plot_feature_correlation_heatmap(df):
    corr = df.select_dtypes(include=["float64","int64"]).corr()
    plt.figure(figsize=(12,8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()

def plot_price_by_borough(df):
    plt.figure(figsize=(10,6))
    sns.boxplot(data=df, x="neighbourhood_group_cleansed", y="price")
    plt.title("Price Distribution by Borough")
    plt.xlabel("Borough")
    plt.ylabel("Price ($)")
    plt.ylim(0,1000)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

def plot_top_correlated_features(df, top_n=5):
    corr = df.select_dtypes(include=["float64","int64"]).corr()["price"].drop("price").abs().nlargest(top_n)
    plt.figure(figsize=(8,5))
    corr.plot(kind="bar", color="teal", edgecolor="black")
    plt.title("Top Features Correlated with Price")
    plt.ylabel("Absolute Correlation")
    plt.grid(True)
    plt.show()

# Load trained model and processed data
def load_model_and_data(pickle_path="model_and_data.pkl"):
    data = joblib.load(pickle_path)
    return (
        data['rf'], data['X_train'], data['X_test'],
        data['y_train'], data['y_test'], data['df']
    )

# Post-model visualizations
def plot_actual_vs_predicted(model, X_test, y_test):
    preds = model.predict(X_test)
    plt.figure(figsize=(6,6))
    plt.scatter(np.expm1(y_test), np.expm1(preds), alpha=0.5)
    m, M = np.expm1(y_test).min(), np.expm1(y_test).max()
    plt.plot([m,M],[m,M],'r--')
    plt.xlabel("Actual Price ($)")
    plt.ylabel("Predicted Price ($)")
    plt.title("Actual vs. Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    idx = np.argsort(importances)[-20:]
    plt.figure(figsize=(8,8))
    plt.barh(feature_names[idx], importances[idx])
    plt.xlabel("Importance")
    plt.title("Top 20 Feature Importances")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_learning_curve_model(model, X_train, y_train):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5,
        scoring="neg_mean_squared_error",
        train_sizes=np.linspace(0.1,1.0,5), n_jobs=-1,
        shuffle=True, random_state=1234
    )
    train_rmse = np.sqrt(-train_scores.mean(axis=1))
    val_rmse   = np.sqrt(-val_scores.mean(axis=1))
    plt.plot(train_sizes, train_rmse, 'o-', label="Train")
    plt.plot(train_sizes, val_rmse, 'o-', label="Validation")
    plt.xlabel("Training Set Size")
    plt.ylabel("RMSE (log-price)")
    plt.legend()
    plt.title("Learning Curve")
    plt.tight_layout()
    plt.show()

def plot_price_distribution(df):
    raw = np.expm1(df['log_price'])
    fig, axes = plt.subplots(1,2,figsize=(14,5))

    # Raw price histogram
    sns.histplot(raw, bins=50, kde=True, ax=axes[0])
    axes[0].set(
        title="Raw Price ($)",
        xlabel="Price ($)",
        ylabel="Count"
    )
    # Log-transformed price histogram
    sns.histplot(df['log_price'], bins=50, kde=True, ax=axes[1])
    axes[1].set(
        title="Log-Transformed Price",
        xlabel="Log(1 + Price)",
        ylabel="Count"
    )
    plt.tight_layout()
    plt.show()

def plot_cost_surface(feature, X_train, y_train, w_min, w_max):
    x = X_train[feature].values
    y = y_train.values
    def mse(w,b): return mean_squared_error(y, w*x + b)
    W,B = np.meshgrid(
        np.linspace(w_min,w_max,100),
        np.linspace(y.min(),y.max(),100)
    )
    Z = np.vectorize(mse)(W,B)
    fig = go.Figure()
    fig.add_trace(go.Surface(x=W,y=B,z=Z,colorscale='Plasma',showscale=True))
    fig.update_layout(
        title=f"MSE Cost Surface on {feature} → log_price",
        scene=dict(xaxis_title='slope (w)',yaxis_title='intercept (b)',zaxis_title='MSE'),
        width=800,height=600
    )
    fig.show()

def plot_mse_by_room_type(model, X_test, y_test):
    preds = model.predict(X_test)
    df_res = X_test.copy()
    df_res['actual'] = np.expm1(y_test)
    df_res['pred']   = np.expm1(preds)
    cols = [c for c in X_test.columns if c.startswith('room_type_')]
    df_res['room_type'] = X_test[cols].idxmax(axis=1).str.replace('room_type_','')
    mse_by = df_res.groupby('room_type').apply(
        lambda d: mean_squared_error(d['actual'],d['pred'])
    ).sort_values()
    plt.figure(figsize=(8,4))
    sns.barplot(x=mse_by.index,y=mse_by.values,palette='mako')
    plt.xlabel('Room Type')
    plt.ylabel('MSE ($²)')
    plt.title('MSE by Room Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_interactive_scatter_matrix(X_test, y_test, preds):
    df_res = X_test.copy()
    df_res['actual']    = np.expm1(y_test)
    df_res['predicted'] = np.expm1(preds)
    df_res['error']     = df_res['predicted'] - df_res['actual']
    cols = [c for c in X_test.columns if c.startswith('room_type_')]
    room_types = X_test[cols].idxmax(axis=1).str.replace('room_type_','')
    dims = ['amenity_count','number_of_reviews','bedrooms','bathrooms','error']
    fig = px.scatter_matrix(df_res,dimensions=dims,color=room_types)
    fig.update_layout(width=800,height=800,title='Linked Scatter Matrix')
    fig.show()

def plot_amenities_treemap(df_res, X_test):
    cols = [c for c in standard_amenities if c in X_test.columns]
    melt = (
        df_res.reset_index()[['index','actual']]
        .merge(
            X_test[cols].reset_index().melt(
                id_vars=['index'],value_vars=cols,
                var_name='amenity',value_name='has_amenity'
            ), on='index'
        )
        .query('has_amenity==1')
    )
    summary = (
        melt.groupby('amenity')
        .agg(count=('has_amenity','count'),avg_price=('actual','mean'))
        .reset_index()
    )
    fig = px.treemap(summary,path=['amenity'],values='count',color='avg_price',
                    color_continuous_scale='Viridis',title='Amenities Treemap')
    fig.update_layout(margin=dict(t=50,l=25,r=25,b=25))
    fig.show()

if __name__ == "__main__":
    try:
        rf, X_train, X_test, y_train, y_test, df_proc = load_model_and_data()
    except FileNotFoundError:
        rf, X_train, X_test, y_train, y_test, df_proc = train_for_visualizations()
        joblib.dump({
            "rf":      rf,
            "X_train": X_train,
            "X_test":  X_test,
            "y_train": y_train,
            "y_test":  y_test,
            "df":      df_proc
        }, "model_and_data.pkl")


    df_raw = load_data()
    plot_price_histogram(df_raw)
    plot_scatter_price_vs_rooms(df_raw)
    plot_feature_correlation_heatmap(df_raw)
    plot_price_by_borough(df_raw)
    plot_top_correlated_features(df_raw)

    plot_actual_vs_predicted(rf, X_test, y_test)
    plot_feature_importances(rf, X_train.columns.values)
    plot_learning_curve_model(rf, X_train, y_train)
    plot_price_distribution(df_proc)
    plot_cost_surface("amenity_count", X_train, y_train, -1, 1)
    plot_cost_surface("bedrooms",X_train, y_train, -5, 5)
    plot_mse_by_room_type(rf, X_test,y_test)

    preds = rf.predict(X_test)
    plot_interactive_scatter_matrix(X_test, y_test, preds)

    df_res = X_test.copy()
    df_res["actual"] = np.expm1(y_test)
    plot_amenities_treemap(df_res, X_test)

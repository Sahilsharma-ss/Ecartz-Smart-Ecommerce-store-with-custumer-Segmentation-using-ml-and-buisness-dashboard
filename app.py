

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import mysql.connector
import numpy as np
import pandas as pd
import decimal
from datetime import datetime, date

# FLASK SETUP
app = Flask(__name__)
CORS(app)

# DATABASE CONNECTION
def get_db_connection():
    # TODO:
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="ecommercedb"
    )

# UTIL HELPERS
def to_float(x):
    """Convert Decimal/int/float to float safely."""
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, decimal.Decimal):
        return float(x)
    return x

def today_date(cursor):
    """Ask DB for CURRENT_DATE() so time math uses DB clock."""
    cursor.execute("SELECT CURRENT_DATE() AS today")
    return pd.to_datetime(cursor.fetchone()["today"])

# ROUTES: PAGES
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/segmentation")
def segmentation_page():
    return render_template("segmentation.html")

# ROUTE: PRODUCTS API
@app.route("/api/products", methods=["GET"])
def api_products():
    db, cur = None, None
    try:
        db = get_db_connection()
        cur = db.cursor(dictionary=True)
        cur.execute("SELECT ProductID, Name, Category, Price, StockQty FROM Products")
        rows = cur.fetchall()
        # Convert Decimals to float for JSON
        for r in rows:
            r["Price"] = to_float(r.get("Price", 0))
        return jsonify(rows)
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
    finally:
        if cur: cur.close()
        if db and db.is_connected(): db.close()

#  SCRATCH K-MEANS 
def robust_scale_fit(X):
    """
    Compute median and IQR for each column.
    Returns: medians, iqrs
    """
    med = np.median(X, axis=0)
    q75 = np.percentile(X, 75, axis=0)
    q25 = np.percentile(X, 25, axis=0)
    iqr = q75 - q25
    # avoid divide by zero
    iqr[iqr == 0] = 1.0
    return med, iqr

def robust_scale_transform(X, med, iqr):
    return (X - med) / iqr

def kmeans_plus_plus_init(X, k, rng):
    """
    KMeans++ initialization for better starting points.
    """
    n, d = X.shape
    centers = np.empty((k, d), dtype=float)
    # pick first center randomly
    idx0 = rng.integers(0, n)
    centers[0] = X[idx0]
    # pick remaining centers
    for i in range(1, k):
        # compute distance to nearest chosen center
        dist_sq = np.min(((X[:, None, :] - centers[:i][None, :, :])**2).sum(axis=2), axis=1)
        probs = dist_sq / (dist_sq.sum() + 1e-12)
        next_idx = rng.choice(n, p=probs)
        centers[i] = X[next_idx]
    return centers

def kmeans_fit_predict(X, k=3, max_iter=100, random_state=42):
    """
    Very small, educational K-Means.
    - X: numpy array (n_samples, n_features)
    - k: clusters
    - Returns: labels (n,), centers (k, d)
    """
    rng = np.random.default_rng(random_state)
    centers = kmeans_plus_plus_init(X, k, rng)

    for _ in range(max_iter):
        # 1) Assign
        # distances squared to each center
        dists = np.sum((X[:, None, :] - centers[None, :, :])**2, axis=2)
        labels = np.argmin(dists, axis=1)

        # 2) Recompute centers
        new_centers = np.vstack([
            X[labels == i].mean(axis=0) if np.any(labels == i) else centers[i]
            for i in range(k)
        ])

        # Stop if converged
        if np.allclose(new_centers, centers, atol=1e-6):
            centers = new_centers
            break
        centers = new_centers

    return labels, centers

#  FEATURE ENGINEERING FOR SEGMENTATION
def prepare_customer_dataframe(cur):
    """
    Create a single customer table with basic aggregates.
    Columns we create:
      - total_orders
      - total_spent
      - last_order_date
      - recency_days
      - tenure_days
      - tenure_months
      - freq_per_month
      - avg_order_value
      - recency_score
    """
    cur.execute(
        """
        SELECT
            c.CustomerID, c.Name, c.Email, c.JoinDate,
            COUNT(DISTINCT o.OrderID) AS total_orders,
            COALESCE(SUM(p.Amount), 0) AS total_spent,
            COALESCE(MAX(o.OrderDate), c.JoinDate) AS last_order_date
        FROM Customers c
        LEFT JOIN Orders o ON o.CustomerID = c.CustomerID
        LEFT JOIN Payments p ON p.OrderID = o.OrderID
        GROUP BY c.CustomerID, c.Name, c.Email, c.JoinDate
        ORDER BY c.CustomerID
        """
    )
    raw = pd.DataFrame(cur.fetchall())
    if raw.empty:
        return raw, None

    # Convert numeric types
    for col in ["total_orders", "total_spent"]:
        if col in raw.columns:
            raw[col] = raw[col].apply(to_float)

    # Dates and time-based features
    t0 = today_date(cur)
    raw["JoinDate"] = pd.to_datetime(raw["JoinDate"])
    raw["last_order_date"] = pd.to_datetime(raw["last_order_date"])

    raw["recency_days"] = (t0 - raw["last_order_date"]).dt.days.clip(lower=0)
    raw["tenure_days"]  = (t0 - raw["JoinDate"]).dt.days.clip(lower=0)
    raw["tenure_months"] = (raw["tenure_days"] / 30.0).replace(0, 1.0)

    # Frequency and monetary
    raw["freq_per_month"] = (raw["total_orders"] / raw["tenure_months"]).fillna(0.0)
    # avoid div by zero
    denom = raw["total_orders"].replace(0, np.nan)
    raw["avg_order_value"] = (raw["total_spent"] / denom).fillna(0.0)

    # Recency score: 1 = very recent purchase, 0 = very old
    raw["recency_score"] = np.exp(-raw["recency_days"] / 90.0)

    return raw, t0

def label_active_clusters(active_df, centers_df):
    """
    Assign human-friendly names to 3 active clusters:
      - Frequent Buyer: highest freq_per_month
      - Expensive Buyer: highest avg_order_value
      - New User: highest (recency_score + 1/(1+tenure_days))
    No duplicates: once a label is assigned to a cluster index, skip it for others.
    """
    label_map = {}
    remaining = set(range(len(centers_df)))

    # 1) Frequent Buyer
    idx_freq = centers_df["freq_per_month"].idxmax()
    label_map[idx_freq] = "Frequent Buyer"
    remaining.discard(idx_freq)

    # 2) Expensive Buyer
    idx_exp = centers_df["avg_order_value"].idxmax()
    if idx_exp in remaining:
        label_map[idx_exp] = "Expensive Buyer"
        remaining.discard(idx_exp)
    else:
        idx_exp = centers_df.loc[list(remaining), "avg_order_value"].idxmax()
        label_map[idx_exp] = "Expensive Buyer"
        remaining.discard(idx_exp)

    # 3) New User (recent + short tenure)
    new_score = centers_df["recency_score"] + (1.0 / (1.0 + centers_df["tenure_days"]))
    idx_new = new_score.idxmax()
    if idx_new not in label_map:
        label_map[idx_new] = "New User"
    else:
        if remaining:
            label_map[remaining.pop()] = "New User"

    return label_map

#  SEGMENTATION API (K=3 FOR ACTIVE) + INACTIVE GROUP
@app.route("/api/segmentation", methods=["GET"])
def api_segmentation():
    db, cur = None, None
    try:
        db = get_db_connection()
        cur = db.cursor(dictionary=True)

        df, t0 = prepare_customer_dataframe(cur)
        if df is None or df.empty:
            return jsonify({"success": True, "customers": [], "summary": {}, "k": 0})

        # Inactive = never purchased OR spent 0
        inactive_mask = (df["total_orders"] <= 0) | (df["total_spent"] <= 0)
        inactive = df.loc[inactive_mask].copy()
        active   = df.loc[~inactive_mask].copy()

        # Edge case: no active users
        if active.empty:
            inactive["Segment"] = "Inactive User"
            customers = inactive[["CustomerID","Name","Email","Segment","total_orders","total_spent","avg_order_value","recency_days","tenure_days"]]
            summary = {"Inactive User": int(len(customers))}
            return jsonify({"success": True, "customers": customers.to_dict(orient="records"), "summary": summary, "k": 0})

        # ----- Build feature matrix for KMeans (only active) -----
        feats = ["freq_per_month", "avg_order_value", "recency_score", "tenure_days"]
        X = active[feats].fillna(0.0).to_numpy(dtype=float)

        # Robust scaling (median/IQR)
        med, iqr = robust_scale_fit(X)
        Xs = robust_scale_transform(X, med, iqr)

        # KMeans (scratch): k=3 for active – other 1 group is "Inactive"
        labels, centers_scaled = kmeans_fit_predict(Xs, k=3, max_iter=100, random_state=42)

        # Inverse scale centers back to original units for readable labeling
        centers = centers_scaled * iqr + med
        centers_df = pd.DataFrame(centers, columns=feats)

        # Map cluster index -> label
        label_map = label_active_clusters(active, centers_df)

        # Attach segment names
        active = active.copy()
        active["Cluster"] = labels
        active["Segment"] = active["Cluster"].map(label_map)

        # Inactive label
        inactive = inactive.copy()
        inactive["Segment"] = "Inactive User"

        # Final output table
        out = pd.concat([active, inactive], ignore_index=True)
        out["total_spent"] = out["total_spent"].apply(to_float).round(2)
        out["avg_order_value"] = out["avg_order_value"].apply(to_float).round(2)

        customers = out[[
            "CustomerID","Name","Email","Segment","total_orders","total_spent",
            "avg_order_value","recency_days","tenure_days"
        ]].sort_values(["Segment","CustomerID"])

        # Summary counts
        summary = customers["Segment"].value_counts().to_dict()

        return jsonify({
            "success": True,
            "customers": customers.to_dict(orient="records"),
            "summary": summary,
            "k": 4  # 3 active clusters + 1 inactive group
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
    finally:
        if cur: cur.close()
        if db and db.is_connected(): db.close()

# PLACE ORDER
@app.route('/api/place_order', methods=['POST'])
def place_order():
    """
    Places a new order:
      - Inserts a new record in Orders table (with TotalAmount, Status)
      - Inserts items into OrderItems
      - Updates Products stock
      - Inserts payment record
      - Commits all or rolls back if any error occurs
    """
    data = request.get_json()
    customer_id = data.get('customer_id')
    cart_items = data.get('cart')

    if not customer_id or not cart_items:
        return jsonify({"success": False, "message": "Missing customer ID or cart items."}), 400

    db, cursor = None, None
    try:
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        db.start_transaction()

        # Calculate total amount
        total_amount = sum(float(item['price']) * int(item['quantity']) for item in cart_items)

        # Insert into Orders (ensure TotalAmount + Status fields included)
        order_query = """
            INSERT INTO Orders (CustomerID, OrderDate, TotalAmount, Status)
            VALUES (%s, NOW(), %s, %s)
        """
        cursor.execute(order_query, (customer_id, total_amount, 'Completed'))
        order_id = cursor.lastrowid

        if not order_id:
            raise Exception("Failed to create new order.")

        # Insert items + update stock
        for item in cart_items:
            product_id = int(item['id'])
            quantity = int(item['quantity'])
            price = float(item['price'])

            # 1. Update stock safely
            update_stock_query = """
                UPDATE Products
                SET StockQty = StockQty - %s
                WHERE ProductID = %s AND StockQty >= %s
            """
            cursor.execute(update_stock_query, (quantity, product_id, quantity))
            if cursor.rowcount == 0:
                raise Exception(f"Insufficient stock for Product ID {product_id}. Transaction aborted.")

            # 2️. Insert item
            item_query = """
                INSERT INTO OrderItems (OrderID, ProductID, Quantity, Price)
                VALUES (%s, %s, %s, %s)
            """
            cursor.execute(item_query, (order_id, product_id, quantity, price))

        #  Insert Payment record
        payment_query = """
            INSERT INTO Payments (OrderID, Amount, PaymentMethod, Status, PaymentDate)
            VALUES (%s, %s, %s, %s, NOW())
        """
        cursor.execute(payment_query, (order_id, total_amount, 'Card', 'Success'))

        # Commit if everything succeeded
        db.commit()

        return jsonify({
            "success": True,
            "order_id": order_id,
            "message": f"Order #{order_id} placed successfully! Total: ${total_amount:.2f}"
        }), 200

    except Exception as e:
        print(f"Transaction failed: {e}")
        if db:
            db.rollback()
        return jsonify({"success": False, "message": f"Order failed: {e}"}), 500
    finally:
        if cursor: cursor.close()
        if db and db.is_connected(): db.close()

#Reports
@app.route("/reports")
def reports_page():
    """Serves the analytics dashboard HTML page."""
    return render_template("reports.html")


@app.route("/api/reports", methods=["GET"])
def get_reports():
    """Returns summarized analytics for dashboard visualizations."""
    db, cur = None, None
    try:
        db = get_db_connection()
        cur = db.cursor(dictionary=True)

        #  Top 5 selling products
        cur.execute("""
            SELECT p.Name, SUM(oi.Quantity) AS Sold
            FROM OrderItems oi
            JOIN Products p ON oi.ProductID = p.ProductID
            GROUP BY oi.ProductID
            ORDER BY Sold DESC
            LIMIT 5;
        """)
        top_products = cur.fetchall()

        #  Monthly revenue trend (current year) 
        cur.execute("""
            SELECT DATE_FORMAT(OrderDate, '%b') AS Month, 
                   SUM(TotalAmount) AS Revenue
            FROM Orders
            WHERE YEAR(OrderDate) = YEAR(CURDATE())
            GROUP BY MONTH(OrderDate), Month
            ORDER BY MONTH(OrderDate);
        """)
        monthly_revenue = cur.fetchall()

        # Category-wise sales 
        cur.execute("""
            SELECT p.Category, SUM(oi.Quantity) AS TotalSold
            FROM OrderItems oi
            JOIN Products p ON oi.ProductID = p.ProductID
            GROUP BY p.Category
            ORDER BY TotalSold DESC;
        """)
        category_sales = cur.fetchall()

        return jsonify({
            "success": True,
            "top_products": top_products,
            "monthly_revenue": monthly_revenue,
            "category_sales": category_sales
        })

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
    finally:
        if cur: cur.close()
        if db and db.is_connected(): db.close()

# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

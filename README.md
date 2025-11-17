# 🛒 ECARTZ - E-Commerce DBMS Project

## 📘 Overview
Ecartz is a **Flask + MySQL** based e-commerce management system demonstrating **database design, analytics, and clustering (K-Means)** from scratch.  
It features a clean UI, order processing, customer segmentation, and a reports dashboard.
LINK : https://ecommerce-web-with-customer-segmentation.onrender.com
---

## 🚀 Features
- Add/View Products and Customers
- Place Orders (reduces stock automatically)
- Payment record management
- Customer segmentation using K-Means (Frequent / Inactive / Expensive / New Users)
- Analytics Dashboard (Top Products, Monthly Revenue, Category Sales)
- Flask + MySQL backend with REST APIs

---

## 🧠 Tech Stack
| Layer | Technology |
|-------|-------------|
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Backend | Flask (Python) |
| Database | MySQL (PlanetScale / Railway) |
| Deployment | Render (backend), Vercel (frontend) |

---

## 🏗️ Folder Structure

```text
ecartz_dbms_project/
│
├── app.py
├── requirements.txt
├── database.sql
├── README.md
│
├── templates/
│   ├── index.html
│   ├── segmentation.html
│   └── reports.html
│
├── static/
│   ├── script.js
│   └── styles.css
├── assets/
│   ├── Business_Analytics_UI.pdf
│   ├── customer_segmentation_ui.png
│   ├── er_diagram.jpg
│   └── landing_page_ui.png
│
└── .env

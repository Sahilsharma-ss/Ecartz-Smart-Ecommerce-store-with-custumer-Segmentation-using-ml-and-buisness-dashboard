-- =============================================================
-- Ecartz E-Commerce Database — PostgreSQL schema for Supabase
-- =============================================================
-- How to import:
--   1. Open your Supabase project → SQL Editor
--   2. Paste this entire file and click "Run"
--   3. Then import the data (see "Importing data" section below)
--
-- Importing data from the original MySQL dump:
--   Option A (Recommended): Use the Supabase Table Editor to
--   upload CSV exports from your MySQL database.
--   Option B: Use pgloader (https://pgloader.io/) to migrate
--   directly from MySQL to PostgreSQL/Supabase.
--   Option C: Strip MySQL-specific lines from Ecartz_database.sql
--   (LOCK TABLES, /*!...*/, ENGINE=...) and run the INSERT
--   statements in the Supabase SQL Editor.
-- =============================================================

-- Drop existing tables (in FK-safe order)
DROP TABLE IF EXISTS payments   CASCADE;
DROP TABLE IF EXISTS orderitems CASCADE;
DROP TABLE IF EXISTS orders     CASCADE;
DROP TABLE IF EXISTS products   CASCADE;
DROP TABLE IF EXISTS customers  CASCADE;

-- -----------------------------------------------------------
-- customers
-- -----------------------------------------------------------
CREATE TABLE customers (
    customerid  SERIAL       PRIMARY KEY,
    name        VARCHAR(50)  NOT NULL,
    email       VARCHAR(50)  NOT NULL UNIQUE,
    age         INTEGER      CHECK (age >= 0),
    gender      VARCHAR(10)  CHECK (gender IN ('Male', 'Female', 'Other')),
    joindate    DATE         NOT NULL DEFAULT CURRENT_DATE
);

CREATE INDEX idx_customers_name ON customers (name);

-- -----------------------------------------------------------
-- products
-- -----------------------------------------------------------
CREATE TABLE products (
    productid   SERIAL        PRIMARY KEY,
    name        VARCHAR(100)  NOT NULL,
    category    VARCHAR(50)   NOT NULL,
    price       INTEGER       NOT NULL,
    stockqty    INTEGER       NOT NULL
);

-- -----------------------------------------------------------
-- orders
-- -----------------------------------------------------------
CREATE TABLE orders (
    orderid      SERIAL       PRIMARY KEY,
    customerid   INTEGER      NOT NULL REFERENCES customers (customerid) ON DELETE CASCADE ON UPDATE CASCADE,
    orderdate    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    totalamount  INTEGER      NOT NULL,
    status       VARCHAR(20)  NOT NULL
);

CREATE INDEX idx_orders_customerid ON orders (customerid);
CREATE INDEX idx_orders_orderdate  ON orders (orderdate);

-- -----------------------------------------------------------
-- orderitems
-- -----------------------------------------------------------
CREATE TABLE orderitems (
    orderitemid  SERIAL    PRIMARY KEY,
    orderid      INTEGER   NOT NULL REFERENCES orders   (orderid)   ON DELETE CASCADE ON UPDATE CASCADE,
    productid    INTEGER   NOT NULL REFERENCES products (productid) ON DELETE CASCADE ON UPDATE CASCADE,
    quantity     INTEGER   NOT NULL CHECK (quantity > 0),
    price        INTEGER   NOT NULL CHECK (price >= 0)
);

CREATE INDEX idx_orderitems_order_product ON orderitems (orderid, productid);

-- -----------------------------------------------------------
-- payments
-- -----------------------------------------------------------
CREATE TABLE payments (
    paymentid      SERIAL       PRIMARY KEY,
    orderid        INTEGER      NOT NULL REFERENCES orders (orderid) ON DELETE CASCADE ON UPDATE CASCADE,
    paymentdate    TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP,
    amount         INTEGER      NOT NULL,
    paymentmethod  VARCHAR(20)  NOT NULL,
    status         VARCHAR(20)  NOT NULL
);

CREATE INDEX idx_payments_orderid     ON payments (orderid);
CREATE INDEX idx_payments_paymentdate ON payments (paymentdate);

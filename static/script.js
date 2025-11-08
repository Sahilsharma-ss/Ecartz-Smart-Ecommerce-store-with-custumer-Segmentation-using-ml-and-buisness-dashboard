// static/script.js

const state = {
  allProducts: [],
  filteredProducts: [],
  cart: [],
  selectedCategory: 'All',
};

const API_BASE_URL = window.location.origin;

// DOM
const categoryFilterEl = document.getElementById('category-filter');
const productListEl = document.getElementById('product-list');
const cartItemsEl = document.getElementById('cart-items');
const cartTotalEl = document.getElementById('cart-total');
const checkoutButtonEl = document.getElementById('checkout-button');
const customerIdInputEl = document.getElementById('customer-id');
const messageBoxEl = document.getElementById('message-box');
const emptyCartMessageEl = document.getElementById('empty-cart-message');

// Render filter buttons
function renderCategoryFilter() {
  categoryFilterEl.innerHTML = '';
  const categories = ['All', ...new Set(state.allProducts.map(p => p.Category))].sort();
  categories.forEach(category => {
    const btn = document.createElement('button');
    btn.className = 'filter-btn';
    btn.textContent = category;
    btn.dataset.category = category;
    if (category === state.selectedCategory) btn.classList.add('active');
    categoryFilterEl.appendChild(btn);
  });
}

// Render product cards
function renderProducts() {
  state.filteredProducts = state.selectedCategory === 'All'
    ? state.allProducts
    : state.allProducts.filter(p => p.Category === state.selectedCategory);

  productListEl.innerHTML = '';
  if (state.filteredProducts.length === 0) {
    productListEl.innerHTML = `<p class="empty-message">No products found in this category.</p>`;
  }
  state.filteredProducts.forEach(p => {
    const el = document.createElement('div');
    el.className = 'product-card';
    el.innerHTML = `
      <div class="product-info">
        <h3>${p.Name}</h3>
        <p>Category: ${p.Category}</p>
        <p>Stock: ${p.StockQty}</p>
        <p class="product-price">$${Number(p.Price).toFixed(2)}</p>
      </div>
      <button class="add-to-cart-btn" data-id="${p.ProductID}">Add to Cart</button>
    `;
    productListEl.appendChild(el);
  });
}

// Render cart
function renderCart() {
  cartItemsEl.innerHTML = '';
  let total = 0;

  if (state.cart.length === 0) {
    emptyCartMessageEl.classList.remove('hidden');
  } else {
    emptyCartMessageEl.classList.add('hidden');
  }

  state.cart.forEach(item => {
    const row = document.createElement('div');
    row.className = 'cart-item';
    row.innerHTML = `
      <div class="cart-item-details">
        <h4>${item.name}</h4>
        <p>${item.quantity} x $${item.price.toFixed(2)}</p>
      </div>
      <button class="remove-btn" data-id="${item.id}">X</button>
    `;
    cartItemsEl.appendChild(row);
    total += item.price * item.quantity;
  });

  cartTotalEl.textContent = `$${total.toFixed(2)}`;
  checkoutButtonEl.disabled = state.cart.length === 0;
}

// Helpers
function showMessage(text, ok) {
  messageBoxEl.textContent = text;
  messageBoxEl.classList.remove('hidden','success','error');
  messageBoxEl.classList.add(ok ? 'success' : 'error');
  setTimeout(() => messageBoxEl.classList.add('hidden'), 4000);
}

// Events
productListEl.addEventListener('click', (e) => {
  const btn = e.target.closest('.add-to-cart-btn');
  if (!btn) return;
  const id = parseInt(btn.dataset.id);
  const p = state.allProducts.find(x => x.ProductID === id);
  if (!p) return;
  const exist = state.cart.find(x => x.id === id);
  if (exist) exist.quantity += 1;
  else state.cart.push({ id: p.ProductID, name: p.Name, price: Number(p.Price), quantity: 1 });
  renderCart();
});

cartItemsEl.addEventListener('click', (e) => {
  const btn = e.target.closest('.remove-btn');
  if (!btn) return;
  const id = parseInt(btn.dataset.id);
  state.cart = state.cart.filter(x => x.id !== id);
  renderCart();
});

categoryFilterEl.addEventListener('click', (e) => {
  const btn = e.target.closest('.filter-btn');
  if (!btn) return;
  document.querySelector('.filter-btn.active')?.classList.remove('active');
  btn.classList.add('active');
  state.selectedCategory = btn.dataset.category;
  renderProducts();
});

checkoutButtonEl.addEventListener('click', async () => {
  const customerId = parseInt(customerIdInputEl.value || '0');
  if (!customerId || customerId <= 0) {
    showMessage("Please enter a valid Customer ID.", false);
    return;
  }
  if (state.cart.length === 0) {
    showMessage("Your cart is empty.", false);
    return;
  }
  const payload = {
    customer_id: customerId,
    cart: state.cart.map(i => ({ id: i.id, quantity: i.quantity, price: i.price }))
  };
  try {
    const res = await fetch(`${API_BASE_URL}/api/place_order`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (data.success) {
      showMessage(data.message, true);
      state.cart = [];
      customerIdInputEl.value = '';
      renderCart();
    } else {
      showMessage(data.message || "Order failed.", false);
    }
  } catch (err) {
    console.error(err);
    showMessage("Network error. Is the Flask server running?", false);
  }
});

// Initial load
async function fetchProducts() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/products`);
    state.allProducts = await res.json();
    renderCategoryFilter();
    renderProducts();
  } catch (err) {
    productListEl.innerHTML = `<p class="error-message">Failed to load products.</p>`;
  }
}

document.addEventListener('DOMContentLoaded', () => {
  fetchProducts();
  renderCart();
});

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency

sns.set(style='dark')

# Helper function yang dibutuhkan untuk menyiapkan berbagai dataframe

def create_monthly_order(df) : 
    monthly_orders_df = df.resample(rule='M', on='order_purchase_timestamp').agg({
    "order_id": "nunique",
    "price": "sum"
})
    monthly_orders_df['Year'] = monthly_orders_df.index.year  # Menambahkan kolom tahun
    monthly_orders_df.index = monthly_orders_df.index.strftime('%B')  # Mengubah format order date menjadi Bulan
    monthly_orders_df = monthly_orders_df.reset_index()
    monthly_orders_df.rename(columns={
    "order_purchase_timestamp" : "Month",
    "order_id": "order_count",
    "price": "revenue"
}, inplace=True)
    
    last_12_months_df = monthly_orders_df.head(12)
    return monthly_orders_df


def create_order_items_df(df):
    sum_order_items_df = df.groupby("product_category_name_english").order_item_id.sum().sort_values(ascending=False).reset_index()
    return sum_order_items_df


def create_bystate_df(df):
    state_customer = df.groupby(by="customer_state").customer_id.nunique().sort_values(ascending=False).reset_index()
    state_customer.rename(columns={
        "customer_id": "customer_count"
    }, inplace=True)
    
    return state_customer

def create_rfm_df(df):
    rfm_df = df.groupby(by="customer_id", as_index=False).agg({
        "order_purchase_timestamp": "max", #mengambil tanggal order terakhir
        "order_id": "nunique",
        "price": "sum"
    })
    rfm_df.columns = ["customer_id", "order_purchase_timestamp", "frequency", "monetary"]
    
    rfm_df["order_purchase_timestamp"] = pd.to_datetime(rfm_df["order_purchase_timestamp"]).dt.to_period('M')
    recent_date = df["order_purchase_timestamp"].max().to_period('M')
    rfm_df["recency"] = rfm_df["order_purchase_timestamp"].apply(lambda x: (recent_date - x).n)
    rfm_df.drop("order_purchase_timestamp", axis=1, inplace=True)
    
    return rfm_df

# Load cleaned data
all_data_df = pd.read_csv("all_data.csv")

datetime_columns = ["order_purchase_timestamp", "order_delivered_customer_date"]
all_data_df.sort_values(by="order_purchase_timestamp", inplace=True)
all_data_df.reset_index(inplace=True)

for column in datetime_columns:
    all_data_df[column] = pd.to_datetime(all_data_df[column])

# Filter data
min_date = all_data_df["order_purchase_timestamp"].min()
max_date = all_data_df["order_purchase_timestamp"].max()

with st.sidebar:
    # Date Range Filter
    st.subheader('Date Range Filter')
    min_date = all_data_df['order_purchase_timestamp'].min().date()
    max_date = all_data_df['order_purchase_timestamp'].max().date()
    start_date, end_date = st.date_input(
        label='Rentang Waktu',
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )
    
     # Product Filter (using selectbox)
    st.subheader('Product Filter')
    all_products = ['All Products'] + list(all_data_df["product_category_name_english"].unique())
    selected_product = st.selectbox("Pilih Produk", all_products, index=0)  # Set index to 0 for default selection 

# Convert date inputs to datetime64[ns]
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Apply Filters to DataFrame
if selected_product == 'All Products':
    main_df = all_data_df[
        (all_data_df["order_purchase_timestamp"] >= start_date) & 
        (all_data_df["order_purchase_timestamp"] <= end_date)
    ]
else:
    main_df = all_data_df[
        (all_data_df["order_purchase_timestamp"] >= start_date) & 
        (all_data_df["order_purchase_timestamp"] <= end_date) & 
        (all_data_df["product_category_name_english"] == selected_product)
    ]

# Prepare various DataFrames for display
monthly_orders_df = create_monthly_order(main_df)
sum_order_items_df = create_order_items_df(main_df)
state_customer = create_bystate_df(main_df)
rfm_df = create_rfm_df(all_data_df)  # Use the entire dataset for RFM DataFrame

# Fungsi slice_customer_id untuk memotong customer_id menjadi 3 huruf depan
def slice_customer_id(customer_id):
    return customer_id[:3]

# Menggunakan fungsi slice_customer_id pada kolom customer_id
rfm_df['short_customer_id'] = rfm_df['customer_id'].apply(slice_customer_id)

# plot number of daily orders (2021)
st.header('NNS Store :star:')
st.subheader('Monthly Orders')

col1, col2 = st.columns(2)

with col1:
    total_orders = monthly_orders_df.order_count.sum()
    st.metric("Total orders", value=total_orders)

with col2:
    total_revenue = format_currency(monthly_orders_df.revenue.sum(), "AUD", locale='es_CO') 
    st.metric("Total Revenue", value=total_revenue)

# Menggabungkan tahun dan bulan
monthly_orders_df["Month_Year"] = monthly_orders_df.apply(lambda x: f"{x['Month']} {x['Year']}", axis=1)

# Membuat visualisasi
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(
    monthly_orders_df.index,  # Menggunakan indeks sebagai sumbu x
    monthly_orders_df["order_count"],
    marker='o', 
    linewidth=2,
    color="#90CAF9"
)
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelrotation=45, labelsize=15)  # Mengatur rotasi label sumbu x agar lebih mudah dibaca
ax.set_xticks(monthly_orders_df.index)  # Menetapkan titik-titik pada sumbu x
ax.set_xticklabels(monthly_orders_df["Month_Year"], rotation=45, ha="right")  # Mengatur label pada sumbu x
ax.set_xlabel("Month and Year", fontsize=18)
ax.set_ylabel("Order Count", fontsize=18)
ax.set_title("Monthly Order Count Over Time", fontsize=20)

plt.tight_layout()
st.pyplot(fig)


# Product
st.subheader("Best & Worst Performing Product")

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35, 15))

colors = ["#068DA9", "#D3D3D3", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

sns.barplot(x="order_item_id", y="product_category_name_english", data=sum_order_items_df.head(5), palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("Number of Sales", fontsize=30)
ax[0].set_title("Best Performing Product", loc="center", fontsize=50)
ax[0].tick_params(axis='y', labelsize=35)
ax[0].tick_params(axis='x', labelsize=30)

sns.barplot(x="order_item_id", y="product_category_name_english", data=sum_order_items_df.sort_values(by="order_item_id", ascending=True).head(5), palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("Number of Sales", fontsize=30)
ax[1].invert_xaxis()
ax[1].yaxis.set_label_position("right")
ax[1].yaxis.tick_right()
ax[1].set_title("Worst Performing Product", loc="center", fontsize=50)
ax[1].tick_params(axis='y', labelsize=35)
ax[1].tick_params(axis='x', labelsize=30)

st.pyplot(fig)

# Customer Demographics
st.subheader("Customer Demographics")

# Pilih lima teratas
top_states = state_customer.sort_values(by="customer_count", ascending=False).head(5)

# Tambahkan kategori "Lainnya" untuk menyimpan sisa negara
other_states_count = state_customer["customer_count"].sum() - top_states["customer_count"].sum()
other_states = pd.DataFrame({"customer_state": ["Lainnya"], "customer_count": [other_states_count]})

# Gabungkan lima teratas dengan kategori "Lainnya"
top_and_other = pd.concat([top_states, other_states])

# Visualisasi menggunakan bar chart
fig, ax = plt.subplots(figsize=(12, 6))

# Menetapkan warna untuk lima bar teratas dan "Lainnya"
colors = ["#D3D3D3"] * len(top_and_other["customer_state"])
colors[0] = "#068DA9"  # Warna untuk kategori "Lainnya"

# Menggunakan bar chart
ax.bar(top_and_other["customer_state"], top_and_other["customer_count"], color=colors)

# Menambahkan label dan judul
ax.set_xlabel("Customer State", fontsize=15)
ax.set_ylabel("Number of Customers", fontsize=15)
ax.set_title("Percentage of Customers by States (Top 5 and other)", fontsize=20)

# Menampilkan bar chart menggunakan Streamlit
st.pyplot(fig)

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming rfm_df is already loaded with the necessary data

# Helper function to format currency
def format_currency(value, currency_symbol="AUD"):
    return f"{currency_symbol} {value:,.2f}"

# Streamlit layout
st.subheader("Best Customer Based on RFM Parameters")

# Metrics display
col1, col2, col3 = st.columns(3)

with col1:
    avg_recency = round(rfm_df.recency.mean(), 1)
    st.metric("Average Recency (Month)", value=avg_recency)

with col2:
    avg_frequency = round(rfm_df.frequency.mean(), 2)
    st.metric("Average Frequency", value=avg_frequency)

with col3:
    avg_monetary = format_currency(rfm_df.monetary.mean(), "AUD")
    st.metric("Average Monetary", value=avg_monetary)

# Visualization
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(35, 15))
colors = ["#90CAF9", "#90CAF9", "#90CAF9", "#90CAF9", "#90CAF9"]

# Plotting by Recency
sns.barplot(y="recency", x="short_customer_id", data=rfm_df.sort_values(by="recency", ascending=True).head(5),
            palette=colors, ax=ax[0])
ax[0].set_ylabel(None)
ax[0].set_xlabel("short_customer_id", fontsize=30)
ax[0].set_title("By Recency (Month)", loc="center", fontsize=50)
ax[0].tick_params(axis='y', labelsize=30)
ax[0].tick_params(axis='x', labelsize=35)

# Plotting by Frequency
sns.barplot(y="frequency", x="short_customer_id", data=rfm_df.sort_values(by="frequency", ascending=False).head(5),
            palette=colors, ax=ax[1])
ax[1].set_ylabel(None)
ax[1].set_xlabel("short_customer_id", fontsize=30)
ax[1].set_title("By Frequency", loc="center", fontsize=50)
ax[1].tick_params(axis='y', labelsize=30)
ax[1].tick_params(axis='x', labelsize=35)

# Plotting by Monetary
sns.barplot(y="monetary", x="short_customer_id", data=rfm_df.sort_values(by="monetary", ascending=False).head(5),
            palette=colors, ax=ax[2])
ax[2].set_ylabel(None)
ax[2].set_xlabel("short_customer_id", fontsize=30)
ax[2].set_title("By Monetary", loc="center", fontsize=50)
ax[2].tick_params(axis='y', labelsize=30)
ax[2].tick_params(axis='x', labelsize=35)

st.pyplot(fig)


with st.expander("See explanation"):
    st.write(
        """Analisis RFM: Memahami Perilaku Pelanggan

Analisis RFM (Recency, Frequency, Monetary) membantu memahami kebiasaan pelanggan dengan membagi mereka ke dalam segmen berdasarkan tiga metrik:

1. Recency (Keterkinian)

Rata-rata: 9,5 bulan
Makna: Pelanggan rata-rata melakukan pembelian terakhir 9,5 bulan lalu.
Interpretasi: Nilai recency yang rendah menunjukkan aktivitas pembelian yang lebih baru, menandakan keterlibatan pelanggan yang positif.
2. Frequency (Frekuensi)

Rata-rata: 1,0
Makna: Pelanggan rata-rata melakukan satu pembelian.
Interpretasi: Frekuensi transaksi tergolong rendah. Angka ini bisa diinterpretasikan berbeda tergantung pada konteks bisnis.
3. Monetary (Nilai)

Rata-rata: AUD 143,54
Makna: Pelanggan rata-rata menghabiskan AUD 143,54 per transaksi.
Interpretasi: Nilai monetary yang tinggi menunjukkan bahwa pelanggan di segmen ini cenderung menghabiskan lebih banyak per transaksi.

Kesimpulan:
Segmen ini terdiri dari pelanggan yang melakukan pembelian terakhir 9,5 bulan lalu dengan frekuensi satu kali dan nilai rata-rata AUD 143,54 per transaksi.
Frekuensi pembelian tergolong rendah, namun nilai monetary menunjukkan bahwa pelanggan di segmen ini berpotensi berbelanja dengan nilai tinggi.
        """
    )
st.caption('Copyright © Nadila Nur Sholekah')
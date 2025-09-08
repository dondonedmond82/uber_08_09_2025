import panel as pn
import pandas as pd
import hvplot.pandas
import warnings
import datetime as dt
from prophet import Prophet

warnings.filterwarnings("ignore")
pn.extension('tabulator', 'echarts', 'bokeh')

# =================== CSS for smaller numbers ===================
pn.config.raw_css.append("""
.small-number .bk-root {
    font-size: 0.7em !important;
}
""")

# =================== Load Data ===================
df_uber = pd.read_csv("./data/csv/ncr_ride_bookings.csv")

df_uber = df_uber[['Date', 'Time', 'Booking ID', 'Booking Status', 'Customer ID',   
                   'Vehicle Type', 'Pickup Location', 'Drop Location', 'Ride Distance',     
                   'Driver Ratings', 'Customer Rating', 'Payment Method']]

df_uber_small = df_uber[['Date', 'Booking Status', 'Vehicle Type', 
                         'Pickup Location','Drop Location', 'Ride Distance',     
                         'Driver Ratings', 'Payment Method']]

df_uber['Pickup Datetime'] = pd.to_datetime(df_uber['Date'] + ' ' + df_uber['Time'])
df_uber['Hour'] = df_uber['Pickup Datetime'].dt.hour

# =================== Widgets ===================
status_select = pn.widgets.Select(name="Booking Status", options=["All"] + df_uber['Booking Status'].unique().tolist())
vehicle_select = pn.widgets.Select(name="Vehicle Type", options=["All"] + df_uber['Vehicle Type'].unique().tolist())
hour_slider = pn.widgets.IntRangeSlider(name='Hour of Day', start=0, end=23, value=(0,23), step=1, sizing_mode='stretch_width')

# =================== Filter Function ===================
def filter_data(status, vehicle, hour_range):
    df = df_uber.copy()
    if status != "All":
        df = df[df['Booking Status'] == status]
    if vehicle != "All":
        df = df[df['Vehicle Type'] == vehicle]
    start_hour, end_hour = hour_range
    df = df[(df['Hour'] >= start_hour) & (df['Hour'] <= end_hour)]
    return df

# =================== Metrics ===================
def summary_metrics(status, vehicle, hour_range):
    df = filter_data(status, vehicle, hour_range)
    total_rides = len(df)
    completed = len(df[df['Booking Status']=="Complete"])
    cancelled_driver = len(df[df['Booking Status']=="Cancelled by Driver"])
    cancelled_customer = len(df[df['Booking Status']=="Cancelled by Customer"])
    no_driver = len(df[df['Booking Status']=="No Driver Found"])
    incomplete = len(df[df['Booking Status']=="Incomplete"])
    avg_distance = df['Ride Distance'].mean() if not df.empty else 0
    avg_driver_rating = df['Driver Ratings'].mean() if not df.empty else 0

    numbers = [
        pn.indicators.Number(name="Total Rides", value=total_rides, width=45, height=30),
        pn.indicators.Number(name="Completed", value=completed, width=45, height=30),
        pn.indicators.Number(name="Cancelled by Driver", value=cancelled_driver, width=45, height=30),
        pn.indicators.Number(name="Cancelled by Customer", value=cancelled_customer, width=45, height=30),
        pn.indicators.Number(name="No Driver Found", value=no_driver, width=45, height=30),
        pn.indicators.Number(name="Incomplete", value=incomplete, width=45, height=30),
        pn.indicators.Number(name="Avg Ride Distance", value=round(avg_distance,2), width=45, height=30),
        pn.indicators.Number(name="Avg Driver Rating", value=round(avg_driver_rating,2), width=45, height=30)
    ]

    # return pn.Row(*numbers, sizing_mode='stretch_width', css_classes=['small-number'])
    return pn.Row()

# =================== Charts ===================
def ride_status_chart(status, vehicle, hour_range):
    df = filter_data(status, vehicle, hour_range)
    return df['Booking Status'].value_counts().hvplot.bar(title="Ride Status", rot=45, height=250, responsive=True)

def rides_over_time_chart(status, vehicle, hour_range):
    df = filter_data(status, vehicle, hour_range)
    rides_by_hour = df.groupby('Hour').size()
    return rides_by_hour.hvplot.line(title="Rides by Hour", ylabel="Number of Rides", xlabel="Hour", height=250, responsive=True)

def ride_distance_histogram(status, vehicle, hour_range):
    df = filter_data(status, vehicle, hour_range)
    return df['Ride Distance'].hvplot.hist(bins=20, title="Ride Distance Distribution", height=250, responsive=True)

def scatter_distance_rating(status, vehicle, hour_range):
    df = filter_data(status, vehicle, hour_range)
    return df.hvplot.scatter(x='Ride Distance', y='Driver Ratings', c='Customer Rating', cmap='Viridis',
                             size=40, title="Distance vs Driver Ratings", height=250, responsive=True)

def top_pickup_locations_map(status, vehicle, hour_range):
    df = filter_data(status, vehicle, hour_range)
    top_locations = df['Pickup Location'].value_counts().head(10).reset_index()
    top_locations.columns = ['Pickup Location','Count']
    return top_locations.hvplot.bar(x='Pickup Location', y='Count', rot=45, title="Top 10 Pickup Locations", height=250, responsive=True)

def forecast_future_bookings(status, vehicle, hour_range):
    df = filter_data(status, vehicle, hour_range)
    daily_rides = df.groupby('Date').size().reset_index(name='y')
    daily_rides['ds'] = pd.to_datetime(daily_rides['Date'])

    if len(daily_rides) < 5:
        return pn.pane.Markdown("⚠️ Not enough data to forecast.")

    model = Prophet()
    model.fit(daily_rides[['ds','y']])

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    chart = forecast.hvplot.line(x='ds', y='yhat', label='Forecast', color='orange') * \
            daily_rides.hvplot.line(x='ds', y='y', label='Actual', color='blue')

    return chart.opts(title="📈 Future Client Bookings Forecast", responsive=True, height=300)

def top_places_chart(status, vehicle, hour_range):
    df = filter_data(status, vehicle, hour_range)
    top_pickup = df['Pickup Location'].value_counts().head(5)
    top_drop = df['Drop Location'].value_counts().head(5)

    data = pd.DataFrame({
        'Location': top_pickup.index.tolist() + top_drop.index.tolist(),
        'Count': top_pickup.values.tolist() + top_drop.values.tolist(),
        'Type': ['Pickup']*len(top_pickup) + ['Drop']*len(top_drop)
    })

    return data.hvplot.bar(x='Location', y='Count', by='Type', 
                           title="🚖 Top 5 Pickup & Drop Locations", rot=45,
                           height=300, responsive=True)

def interactive_table():
    return pn.widgets.Tabulator(df_uber_small, pagination='remote', page_size=15, sizing_mode='stretch_width', height=425)

# =================== Page Layout ===================
def create_page1():
    charts_row1 = pn.Row(
        pn.bind(ride_status_chart, status_select, vehicle_select, hour_slider),
        pn.bind(rides_over_time_chart, status_select, vehicle_select, hour_slider),
        sizing_mode='stretch_width'
    )
    charts_row2 = pn.Row(
        pn.bind(ride_distance_histogram, status_select, vehicle_select, hour_slider),
        pn.bind(scatter_distance_rating, status_select, vehicle_select, hour_slider),
        sizing_mode='stretch_width'
    )
    
    return pn.Column(
        pn.Row(status_select, vehicle_select, hour_slider, sizing_mode='stretch_width'),
        pn.bind(summary_metrics, status_select, vehicle_select, hour_slider),
        charts_row1,
        charts_row2,
        pn.bind(top_pickup_locations_map, status_select, vehicle_select, hour_slider),
        pn.bind(forecast_future_bookings, status_select, vehicle_select, hour_slider),
        pn.bind(top_places_chart, status_select, vehicle_select, hour_slider),
        sizing_mode='stretch_width'
    )

def create_page2():
    return pn.Column(
        pn.pane.Markdown("## Detailed Ride Table"),
        interactive_table(),
        sizing_mode='stretch_width'
    )

# =================== Sidebar & Navigation ===================
mapping = {"Dashboard": create_page1(), "Rides Table": create_page2()}

btn_dashboard = pn.widgets.Button(name="Dashboard", button_type="primary", icon="dashboard")
btn_table = pn.widgets.Button(name="Rides Table", button_type="primary", icon="table")
logout = pn.widgets.Button(name="Logout", button_type="danger", icon="sign-out-alt")
logout.js_on_click(code="window.location.href = './logout'")

main_area = pn.Column(mapping['Dashboard'], sizing_mode='stretch_width')

def show_page(page_key):
    main_area.clear()
    main_area.append(mapping[page_key])

btn_dashboard.on_click(lambda event: show_page("Dashboard"))
btn_table.on_click(lambda event: show_page("Rides Table"))

sidebar = pn.Column(
    pn.pane.Markdown("## Menu"), btn_dashboard, btn_table, logout,
    styles={"width":"100%","padding":"15px"}
)

# =================== Template ===================
template = pn.template.BootstrapTemplate(
    title="My Uber Drives - Compact Dashboard",
    sidebar=[sidebar],
    main=[main_area],
    header_background="black",
    sidebar_width=250
)

template.servable()

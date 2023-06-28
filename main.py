import streamlit as st
from datetime import datetime, timedelta, date
import pandas as pd
import os
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import colorlover as cl
import plotly.colors as pc

st.set_page_config(page_title="Express_Customers", page_icon=":bar_chart:", layout="wide")


@st.cache_data()
def load_excel_files(directory):
    excel_files = [os.path.join(directory, f) for f in os.listdir(directory)]
    dfs = []
    for file in excel_files:
        xl = pd.read_excel(file, sheet_name=None)
        for sheet_name, df in xl.items():
            if not df.empty:
                dfs.append(df)

    return pd.concat(dfs).drop_duplicates()


customers = ['New Customer 1', 'New Customer 2', 'New Customer 3', 'New Customer 4', 'New Customer 5', 'New Customer 6', 'New Customer 7', 'New Customer 8', 'New Customer 9', 'New Customer 10']
ontime = 'ontime'
abn = 'abn'
snd = 'snd'

ontime_df = load_excel_files(ontime)
ontime_df['Planned Signing Time'] = pd.to_datetime(ontime_df['Planned Signing Time']).dt.date
abn_df = load_excel_files(abn)
max_date, min_date = ontime_df["Planned Signing Time"].max(), ontime_df["Planned Signing Time"].min()
snd_df = load_excel_files(snd)
snd_df['Pickup Date'] = pd.to_datetime(snd_df['Pickup Date']).dt.date

st.title("Customer Dashboard")
with st.sidebar:
    st.subheader("Filter Options")
    cst = st.selectbox("Customer Name", customers)
    start_date = min_date
    end_date = max_date
    selected_dates = st.select_slider("Select a date range", options=[start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)], value=(start_date, end_date))


def compute_on_time_signing_rates(ontime_df):
    df = ontime_df
    df = df.query('`Sender Customer` == @cst & `Planned Signing Time` >= @selected_dates[0] & `Planned Signing Time` <= @selected_dates[1]')
    pivot = df.groupby(['Planned Signing Time', 'Agency Area Name', 'Delivery Branch Name'])['Time type'].value_counts().unstack().fillna(0).reset_index()
    pivot['Receivable Amount'] = pivot.iloc[:, 3:].sum(axis=1)
    pivot2 = df[(df['Time type'] == 'delayed') & (df['Status'] == 'signed')].groupby(['Planned Signing Time', 'Agency Area Name', 'Delivery Branch Name'])['Time type'].count().reset_index()
    pivot2.rename(columns={"Time type": "Delay Sign Amount"}, inplace=True)
    pivot = pivot.merge(pivot2, on=['Planned Signing Time', 'Agency Area Name', 'Delivery Branch Name'], how='left')
    pivot['Delay Sign Amount'].fillna(0, inplace=True)
    pivot['Sign Rate'] = (pivot['On-time'] + pivot['Delay Sign Amount']) / pivot['Receivable Amount']
    pivot.rename(columns={"On-time": "On-time signing Amount", "Planned Signing Time": "Date"}, inplace=True)
    ontime_df = pivot
    ontime_df["On Time Signing Rate"] = ontime_df["On-time signing Amount"] / ontime_df["Receivable Amount"]
    ontime_df = ontime_df[["Date", "Agency Area Name", "Delivery Branch Name", "Receivable Amount", "On Time Signing Rate", 'Sign Rate', 'On-time signing Amount', 'Delay Sign Amount']]
    ontime_ag = pivot.groupby(['Date', 'Agency Area Name'])[["On-time signing Amount", 'Delay Sign Amount', "Receivable Amount"]].sum().reset_index()
    ontime_ag["On Time Signing Rate"] = ontime_ag["On-time signing Amount"] / ontime_ag["Receivable Amount"]
    ontime_ag['Sign Rate'] = (ontime_ag['On-time signing Amount'] + ontime_ag['Delay Sign Amount']) / ontime_ag['Receivable Amount']
    ontime_ag = ontime_ag[["Date", "Agency Area Name", "Receivable Amount", "On Time Signing Rate", 'Sign Rate', 'On-time signing Amount', 'Delay Sign Amount']]
    ontime_total = pivot.groupby(['Date'])[['On-time signing Amount', 'Receivable Amount', 'Delay Sign Amount']].sum().reset_index()
    ontime_total["On Time Signing Rate"] = ontime_total["On-time signing Amount"] / ontime_total["Receivable Amount"]
    ontime_total['Sign Rate'] = (ontime_total['On-time signing Amount'] + ontime_total['Delay Sign Amount']) / ontime_total['Receivable Amount']
    ontime_total = ontime_total[["Date", 'On-time signing Amount', 'Delay Sign Amount', "Receivable Amount", "On Time Signing Rate", 'Sign Rate']]
    ontime_total2 = ontime_total
    ontime_total2["k"] = 'k'
    ontime_total2 = ontime_total2.groupby(['k'])[['On-time signing Amount', 'Receivable Amount', 'Delay Sign Amount']].sum().reset_index()
    return ontime_df, ontime_ag, ontime_total, ontime_total2


def over_capacity(ontime_df, abn_df):
    ontime_df = ontime_df
    abn_df = abn_df
    ontime_df = ontime_df.query('`Sender Customer` == @cst & `Planned Signing Time` >= @selected_dates[0] & `Planned Signing Time` <= @selected_dates[1]')
    ontime_df['Date'] = pd.to_datetime(ontime_df['Planned Signing Time'], errors='coerce').dt.date
    abn_df['Date'] = pd.to_datetime(abn_df['Registered time'], errors='coerce').dt.date

    ontime_df['Branch'] = ontime_df['Delivery Branch Name']
    abn_df['Branch'] = abn_df['Registered Branch']

    abn_df = abn_df.sort_values(by="Registered time", ascending=False).drop_duplicates(subset=["Waybill NO.", "Date"])
    abn_df = abn_df[['Waybill NO.', 'Sub-reason', 'Branch', 'Registration Center', 'Registered time', "Date"]]
    ontime_df = ontime_df[["Waybill NO.", "Agency Area Name", "Branch", "Time type", "Planned Signing Time", "Date", "Sender Customer"]]
    ontime_df2 = ontime_df[ontime_df["Time type"] == "delayed"]

    merged_df = pd.merge(ontime_df2, abn_df, how="left", on=["Waybill NO.", "Branch"])

    merged_df.loc[merged_df["Date_y"] <= merged_df["Date_x"], "Sub-reason_x"] = merged_df["Sub-reason"]

    result_df = merged_df.loc[merged_df["Sub-reason_x"].notna()].sort_values(by="Registered time", ascending=False).drop_duplicates(subset=["Waybill NO.", "Branch"])
    result_df = ontime_df2.merge(result_df[["Waybill NO.", "Sub-reason_x"]], on="Waybill NO.", how="left")
    total_df = \
        result_df[(result_df["Sub-reason_x"] == "Holding in Branch") | (result_df["Sub-reason_x"] == "Transportation capacity shortage") | (result_df["Sub-reason_x"] == "Transport in The Next Trip") | (result_df["Sub-reason_x"].isna())].groupby(
            'Date')[
            "Waybill NO."].count().reset_index()
    total_df.rename(columns={"Waybill NO.": "Over-Capacity Shipments"}, inplace=True)

    total_df2 = result_df[(result_df["Sub-reason_x"] == "Holding in Branch") | (result_df["Sub-reason_x"] == "Transportation capacity shortage") | (result_df["Sub-reason_x"] == "Transport in The Next Trip") | (result_df["Sub-reason_x"].isna())]
    total_df2 = total_df2["Waybill NO."].count()

    ag_df = result_df[(result_df["Sub-reason_x"] == "Holding in Branch") | (result_df["Sub-reason_x"] == "Transportation capacity shortage") | (result_df["Sub-reason_x"] == "Transport in The Next Trip") | (result_df["Sub-reason_x"].isna())].groupby(
        ['Date', 'Agency Area Name'])[
        "Waybill NO."].count().reset_index()
    ag_df.rename(columns={"Waybill NO.": "Over-Capacity Shipments"}, inplace=True)

    br_df = result_df[(result_df["Sub-reason_x"] == "Holding in Branch") | (result_df["Sub-reason_x"] == "Transportation capacity shortage") | (result_df["Sub-reason_x"] == "Transport in The Next Trip") | (result_df["Sub-reason_x"].isna())].groupby(
        ['Date', 'Agency Area Name', 'Branch'])[
        "Waybill NO."].count().reset_index()
    br_df.rename(columns={"Waybill NO.": "Over-Capacity Shipments"}, inplace=True)
    return result_df, total_df, ag_df, br_df, total_df2


ontime_br, ontime_ag, ontime_total, ontime_total2 = compute_on_time_signing_rates(ontime_df)
result_df, total_df, ag_df, br_df, total_df2 = over_capacity(ontime_df, abn_df)
snd_df = snd_df.query('`Client Name` == @cst')
# Convert pickup date to datetime and calculate the differences in days
snd_df['Pickup Date'] = pd.to_datetime(snd_df['Pickup Date'], errors='coerce')
snd_df['Days Elapsed'] = (pd.Timestamp(date(2023, 6, 15)) - snd_df['Pickup Date']).dt.days


# Filter the data based on different time durations
exceeded_3_days = snd_df.loc[(snd_df['Days Elapsed'] > 3) & (snd_df['Days Elapsed'] <= 5), 'Waybill NO.']
exceeded_5_days = snd_df.loc[(snd_df['Days Elapsed'] > 5) & (snd_df['Days Elapsed'] <= 7), 'Waybill NO.']
exceeded_7_days = snd_df.loc[(snd_df['Days Elapsed'] > 7) & (snd_df['Days Elapsed'] <= 10), 'Waybill NO.']
exceeded_10_days = snd_df.loc[(snd_df['Days Elapsed'] > 10) & (snd_df['Days Elapsed'] <= 15), 'Waybill NO.']
exceeded_15_days = snd_df.loc[snd_df['Days Elapsed'] > 15, 'Waybill NO.']
within_3_days = snd_df.loc[snd_df['Days Elapsed'] <= 3, 'Waybill NO.']
br_df["Delivery Branch Name"] = br_df['Branch']

forline = ontime_total.merge(total_df, on=["Date"], how='left')
forline["On-Time Delivery"] = (forline["Receivable Amount"] - forline['Over-Capacity Shipments']) / forline['Receivable Amount']

rates_br = ontime_br.merge(br_df, how='left', on=['Date', 'Agency Area Name', 'Delivery Branch Name'])
rates_br = rates_br.groupby(['Agency Area Name', 'Delivery Branch Name'])[['On-time signing Amount', 'Receivable Amount', 'Delay Sign Amount', 'Over-Capacity Shipments']].sum().reset_index()
rates_br["On-Time Delivery"] = (rates_br["Receivable Amount"] - rates_br["Over-Capacity Shipments"]) / rates_br["Receivable Amount"]
rates_br["On-Time Sign"] = rates_br["On-time signing Amount"] / rates_br["Receivable Amount"]
rates_br["Sign Rate"] = (rates_br["On-time signing Amount"] + rates_br["Delay Sign Amount"]) / rates_br["Receivable Amount"]

rates_ag = ontime_ag.merge(ag_df, how='left', on=['Date', 'Agency Area Name'])
rates_ag = rates_ag.groupby(['Agency Area Name'])[['On-time signing Amount', 'Receivable Amount', 'Delay Sign Amount', 'Over-Capacity Shipments']].sum().reset_index()
rates_ag["On-Time Delivery"] = (rates_ag["Receivable Amount"] - rates_ag["Over-Capacity Shipments"]) / rates_ag["Receivable Amount"]
rates_ag["On-Time Sign"] = rates_ag["On-time signing Amount"] / rates_ag["Receivable Amount"]
rates_ag["Sign Rate"] = (rates_ag["On-time signing Amount"] + rates_ag["Delay Sign Amount"]) / rates_ag["Receivable Amount"]

st.markdown("---")
one, two, three, four, five = st.columns(5)
one.metric('On-Time Sign', "{:.2%}".format(ontime_total2.iloc[0, 1] / ontime_total2.iloc[0, 2]))

two.metric('On-Time Delivery', "{:.2%}".format((ontime_total2.iloc[0, 2] - total_df2) / ontime_total2.iloc[0, 2]))
if two.button("View", key=2):
    two.write('There was {:.0f} Over Capacity Shipments and {:.0f} Receivable During The Selected Period'.format(total_df2, ontime_total2.iloc[0, 2]))
    two.write("\n".join(
        result_df[(result_df["Sub-reason_x"] == "Holding in Branch") | (result_df["Sub-reason_x"] == "Transportation capacity shortage") | (result_df["Sub-reason_x"] == "Transport in The Next Trip") | (result_df["Sub-reason_x"].isna())][
            "Waybill NO."].to_list()))

three.metric('Sign Rate', "{:.2%}".format((ontime_total2.iloc[0, 1] + ontime_total2.iloc[0, 3]) / ontime_total2.iloc[0, 2]))

four.metric('Unsigned', snd_df.loc[snd_df['Signing Status'] == 'Unsigned', 'Waybill NO.'].count())
if four.button("View", key=3):
    q, w, e, r, t, y = st.columns(6)
    q.metric("Within 3 Days Since Pickup", within_3_days.count())

    w.metric("Exceeded 3 Days Since Pickup", exceeded_3_days.count())

    e.metric("Exceeded 5 Days Since Pickup", exceeded_5_days.count())
    e.write("\n".join(exceeded_5_days.to_list()))

    r.metric("Exceeded 7 Days Since Pickup", exceeded_7_days.count())
    r.write("\n".join(exceeded_7_days.to_list()))

    t.metric("Exceeded 10 Days Since Pickup", exceeded_10_days.count())
    t.write("\n".join(exceeded_10_days.to_list()))

    y.metric("Exceeded 15 Days Since Pickup", exceeded_15_days.count())
    y.write("\n".join(exceeded_15_days.to_list()))

five.metric('Total Receivable', "{:.0f}".format(ontime_total2.iloc[0, 2]))

st.markdown("---")
color_palette = px.colors.qualitative.Set1

line_styles = {
    "On Time Signing Rate": dict(line=dict(dash="solid", color=color_palette[0])),
    "Sign Rate": dict(line=dict(dash="solid", color=color_palette[1])),
    "On-Time Delivery": dict(line=dict(dash="solid", color=color_palette[2]))
}

fig = go.Figure()

# Add traces for On-Time Signing Rate, Sign Rate, and On-Time Delivery
for column in ["On Time Signing Rate", "Sign Rate", "On-Time Delivery"]:
    fig.add_trace(
        go.Scatter(
            x=forline["Date"],
            y=forline[column],
            mode="lines+markers",
            name=column,
            line=line_styles[column]["line"],
            marker=dict(size=8, line=dict(width=1, color="white")),
            hovertemplate="Date: %{x|%b %d, %Y (%a)}<br>%{yaxis.title.text}: %{y:.2%}",
            hoverinfo="text"
        )
    )

# Configure layout
fig.update_layout(
    title="Signing Rates and On-Time Delivery Over Time",
    xaxis=dict(title="Date", gridcolor="lightgray", tickfont=dict(size=12)),
    yaxis=dict(title="Rates", gridcolor="lightgray", tickformat=".0%", tickfont=dict(size=12), range=[0, 1]),
    margin=dict(l=50, r=50, t=100, b=50),
    template="plotly_white",
)

# Show the figure
onee, twoo = st.columns(2)
twoo.plotly_chart(fig, use_container_width=True)
twoo.write(
    rates_ag[["Agency Area Name", "On-Time Sign", "On-Time Delivery", "Sign Rate", "Receivable Amount", "Over-Capacity Shipments"]].sort_values(by="On-Time Sign", ascending=False)
    .set_index("Agency Area Name")
    .style.format({
        "On-Time Sign": "{:.2%}",
        "On-Time Delivery": "{:.2%}",
        "Sign Rate": "{:.2%}",
        "Receivable Amount": "{:.0f}",
        "Over-Capacity Shipments": "{:.0f}"

    })
)
ag_filter = twoo.radio('Choose Agency Name:', np.append("", ontime_df["Agency Area Name"].sort_values().unique()), horizontal=True)
if ag_filter != "":
    with twoo.expander("Details", expanded=True):
        rates_br = rates_br.loc[(rates_br['Agency Area Name'] == ag_filter)]
        st.write(
            rates_br[["Delivery Branch Name", "On-Time Sign", "On-Time Delivery", "Sign Rate", "Receivable Amount", "Over-Capacity Shipments"]].sort_values(by="On-Time Sign", ascending=False)
            .set_index("Delivery Branch Name")
            .style.format({
                "On-Time Sign": "{:.2%}",
                "On-Time Delivery": "{:.2%}",
                "Sign Rate": "{:.2%}",
                "Receivable Amount": "{:.0f}",
                "Over-Capacity Shipments": "{:.0f}"

            })
        )

# Calculate the percentage of each subreason
abn_df['Date'] = pd.to_datetime(abn_df['Registered time'], errors='coerce').dt.date
abn_df = abn_df.query('`Client Name` == @cst & `Date` >= @selected_dates[0] & `Date` <= @selected_dates[1]')

subreason_counts = abn_df['Sub-reason'].value_counts()
subreason_percentages = subreason_counts / subreason_counts.sum()

# Create a DataFrame with subreason percentages
subreason_df = pd.DataFrame({
    'Subreasons': subreason_percentages.index,
    'Percentage': subreason_percentages.values
})
subreason_df = subreason_df.sort_values('Subreasons')

# Get the number of unique Subreasons
num_subreasons = len(subreason_df['Subreasons'].unique())

# Define a color palette
base_palette = cl.scales['12']['qual']['Set3']
color_palette = cl.interp(base_palette, num_subreasons)

# Create the pie chart
fig = px.pie(
    data_frame=subreason_df,
    values='Percentage',
    names='Subreasons',
    title='Percentage of Abnormals',
    color='Subreasons',
    color_discrete_sequence=color_palette,
    hole=0.4,
    labels={'Percentage': 'Percentage', 'Subreasons': 'Subreasons'},
    hover_data=['Percentage'],
)

# Customize the layout
fig.update_traces(textposition='inside', textinfo='percent+label', hovertemplate='%{label}:<br>%{percent:.1%}')
fig.update_layout(
    margin=dict(l=20, r=20, t=60, b=20),
    plot_bgcolor='white'
)

# Show the pie chart
onee.plotly_chart(fig, use_container_width=True)

# Calculate the percentage of each subreason for each agency
agency_subreason_counts = abn_df.groupby(['Registration Center', 'Sub-reason']).size().unstack()
agency_subreason_percentages = agency_subreason_counts.div(agency_subreason_counts.sum(axis=1), axis=0)

# Reshape the data for the stacked bar chart
agency_subreason_df = agency_subreason_percentages.reset_index().melt(id_vars='Registration Center', var_name='Subreasons', value_name='Percentage')
# Sort the DataFrame alphabetically by 'Subreasons'
agency_subreason_df = agency_subreason_df.sort_values('Subreasons')
# Create the stacked bar chart
fig = px.bar(
    data_frame=agency_subreason_df,
    x='Registration Center',
    y='Percentage',
    color='Subreasons',
    title='Subreason Percentages by Agency',
    color_discrete_sequence=color_palette,
    labels={'Percentage': 'Percentage', 'Registration Center': 'Agency Name'},
    hover_data=['Percentage'],
    barmode='stack'
)

# Customize the layout
fig.update_traces(textposition='inside', texttemplate='%{y:.1%}')
fig.update_layout(
    xaxis=dict(title='Agency Name'),
    yaxis=dict(title='Percentage', range=[0, 1]),
    margin=dict(l=40, r=40, t=60, b=20),
    plot_bgcolor='white'
)

# Show the stacked bar chart
onee.plotly_chart(fig, use_container_width=True)

ag_filter2 = onee.radio('Choose Agency Name: ', np.append("", ontime_df["Agency Area Name"].sort_values().unique()), horizontal=True)
if ag_filter2 != "":
    with st.expander("Details ", expanded=True):
        abn_df = abn_df.loc[abn_df["Registration Center"] == ag_filter2]
        # Calculate the percentage of each subreason for each agency
        branch_subreason_counts = abn_df.groupby(['Registered Branch', 'Sub-reason']).size().unstack()
        branch_subreason_counts = branch_subreason_counts.div(branch_subreason_counts.sum(axis=1), axis=0)

        # Reshape the data for the stacked bar chart
        branch_subreason_df = branch_subreason_counts.reset_index().melt(id_vars='Registered Branch', var_name='Subreasons', value_name='Percentage')
        # Sort the DataFrame alphabetically by 'Subreasons'
        branch_subreason_df = branch_subreason_df.sort_values('Subreasons')
        # Create the stacked bar chart
        fig = px.bar(
            data_frame=branch_subreason_df,
            x='Registered Branch',
            y='Percentage',
            color='Subreasons',
            title=f'Subreason Percentages by Branch in {ag_filter2}',
            color_discrete_sequence=color_palette,
            labels={'Percentage': 'Percentage'},
            hover_data=['Percentage'],
            barmode='stack'
        )

        # Customize the layout
        fig.update_traces(textposition='inside', texttemplate='%{y:.1%}')
        fig.update_layout(
            xaxis=dict(title='Registered Branch'),
            yaxis=dict(title='Percentage', range=[0, 1]),
            margin=dict(l=40, r=40, t=60, b=20),
            plot_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)

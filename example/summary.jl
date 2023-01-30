using Pandas
using PlotlyJS
using Dates
using PyPlot

dataf = Pandas.DataFrame(df)
Pandas.describe(dataf)


timestamp = convert(Array{DateTime},timestamp)

PlotlyJS.plot(
    timestamp, df,
    Layout(
        title="Time Series with Range Slider and Selectors",
        xaxis=attr(
            rangeslider_visible=true,
            rangeselector=attr(
                buttons=[
                    attr(count=1, label="1d", step="day", stepmode="backward"),
                    attr(count=1, label="1w", step="week", stepmode="backward"),
                    attr(count=1, label="1m", step="month", stepmode="backward"),
                    attr(count=6, label="6m", step="month", stepmode="backward"),
                    attr(count=1, label="1y", step="year", stepmode="backward"),  
                ]
            )
        )
    )
)

#to be continued
#add here some other nice plots...
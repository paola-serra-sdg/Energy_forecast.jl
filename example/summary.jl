using Pandas
using PlotlyJS
using Dates
using PyPlot
using TimeSeries
using StatsPlots


#summary statistics for trained user
dataf1 = Pandas.DataFrame(df_single)
d = Pandas.describe(dataf1)
d



#summary statistic for all users
dframe = Array{Any}(undef, numfiles)
d = Array{Any}(undef, numfiles)
for i in 1:numfiles 
    dframe[i] = Pandas.DataFrame(df[i])
    d[i] = Pandas.describe(dframe[i])   
end



timestamp = convert(Array{DateTime},timestamp)


Plots.PlotlyJSBackend()
#plot active power demand for  1st user 

PlotlyJS.plot(
    timestamp, df_single,alpha = 0.4,lw = 10,
    Layout(
        title="Active Power Demand for user 1358568",
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

using Plots


trace1 = PlotlyJS.box(;y = df[9], name = "1352954")
PlotlyJS.plot(trace1,
    Layout(
        title="Box Plot for user 1352954",
        xaxis=attr(
            rangeslider_visible=true)
))

user = Vector{Any}(undef, numfiles)
for i in 1:numfiles
    user[i] = files[i][end-11:end-5] 
end




data = GenericTrace[]
for i in 1:7
    trace = PlotlyJS.box(;y= df[i],
                 name=files[i][end-11:end-5] )
    push!(data, trace)
end

t = "Box Plot for each user"
layout = Layout(;title=t,xaxis=attr(attr(rangeslider_visible=true),
                    #rangeselector=attr(
                                    #buttons=[
                                       # attr(count=1, label="1d", step="day", stepmode="backward")],
                    
                     yaxis=attr(;zeroline=false, gridcolor="white"),
                     paper_bgcolor="rgb(233, 233, 233)",
                     plot_bgcolor="rgb(233, 233, 233)"))#,
                     #showlegend=true))

PlotlyJS.plot(data, layout)





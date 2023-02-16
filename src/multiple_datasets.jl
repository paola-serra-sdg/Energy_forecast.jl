#WORKING IN PROGESS
#do not check thic code
#read all files


numfiles = numfiles-2
#initialize vector 
tempdfs = Vector{Any}(undef, numfiles)
dfs = Vector{Any}(undef, numfiles)
sheet = Vector{Any}(undef, numfiles)
s = Vector{String}(undef, numfiles)
df = Vector{Any}(undef, numfiles)
df_st = Vector{Any}(undef, numfiles)
x_train = Vector{Any}(undef, numfiles)
y_train = Vector{Any}(undef, numfiles)
x_test = Vector{Any}(undef, numfiles)
y_test = Vector{Any}(undef, numfiles)


train_data = Vector{Any}(undef, numfiles)
test_data = Vector{Any}(undef, numfiles)

for i in 1:numfiles #-2
    dfs[i]= XLSX.readxlsx(files[i])  
    sheet[i] = dfs[i][" ACTIVA Y REACTIVA"]
    s[i] = string(size(sheet[i][:],1))
    df[i] = sheet[i]["F4:F"*s[i]]
    df[i] = string.(df[i])
    df[i] = parse.(Float64, df[i])
    df[i] = Vector{Float64}(vec(df[i]))  
    df_st[i] = standardize(df[i])
    println(i, " done")
end


user = Vector{Any}(undef, numfiles)
for i in 1:numfiles
    user[i] = files[i][end-11:end-5] 
end
#remove file with size less than the end of the 44th week

count_removed_files = 0
j = 1
for i in 1:numfiles
    if parse.(Float64,s[i]) >= 30240
        df_st[j] = df_st[i]
        user[j] = user[i]
        s[j] = s[i]
        j += 1
    else
        println("Removing file: ", user[i], " with size: ", s[i])
        count_removed_files += 1
    end
end
println("Number of files removed: ", count_removed_files)
#update numfiles 

numfiles = numfiles -  count_removed_files 
user = user[1:numfiles]
df_st = df_st[1:numfiles,:,:]


week = 44


for i in 1:numfiles
    x_train[i] = df_st[i][1:start_week_prediction, :, :]
    y_train[i] = df_st[i][week_length:end, :, :]
    println("train of ",i," done")
    x_test[i] = df_st[i][start_week_prediction+1:end_week_prediction,:,:]
    y_test[i] = df_st[i][start_week_prediction+1:end_week_prediction,:,:]
    println("test of ",i," done")
    train_data[i] = DataLoader((x_train[i], y_train[i]) ; batchsize = 1)
    test_data[i] = DataLoader((x_test[i], y_test[i]); batchsize = 1)
    println("wrap of ",i," done")
end

for i in 1:numfiles
    y_test[i] = df_st[i][start_week_prediction+1:end_week_prediction,:,:]
    println("y_test of ",i," done")
end


ŷ_PM_multi = Vector{Any}(undef, numfiles)
ŷ_CNN_multi = Vector{Any}(undef, numfiles)
y_multi = Vector{Any}(undef, numfiles)

for i in 1:numfiles
    ŷ_PM_multi[i] = model_PM_adam(y_test[i])[:]
    ŷ_CNN_multi[i] = model_CNN_adam(y_test[i])[:]
    y_multi[i] = y_test[i][:] #ground truth
end






##training for each Users
N = 24 * 4
model_CNN_multi = Vector{Any}(undef, numfiles)
params_CNN_multiple = Vector{Any}(undef, numfiles)



for i in 1:numfiles
    model_CNN_multi[i] = Chain(Conv((N,),1 => 1, pad = (N-1,0),sigmoid),
                            Conv((N,),1 => 1, pad = (N-1,0),sigmoid),
                            Conv((N,),1 => 1, pad = (N-1,0),sigmoid),
                            Conv((N,),1 => 1, pad = (N-1,0),sigmoid),
                            Conv((N,),1 => 1, pad = (N-1,0),sigmoid),
                            Conv((N,),1 => 1, pad = (N-1,0),sigmoid),
                            Conv((N,),1 => 1, pad = (N-1,0),sigmoid),
                            Conv((N,),1 => 1, pad = (N-1,0),sigmoid),
                            Conv((N,),1 => 1, pad = (N-1,0)),
                            Conv((N,),1 => 1, pad = (N-1,0)),
                            Conv((N,),1 => 1, pad = (N-1,0)),
                            Conv((N,),1 => 1, pad = (N-1,0)),
                            Conv((N,),1 => 1, pad = (N-1,0)))    |> f64;
                            #Dense(32925, 672)) 

    params_CNN_multiple[i] = Flux.params(model_CNN_multi[i]);
end

losses = Vector{Any}(undef, numfiles)

losses = (x, y) -> Flux.Losses.mse(model_CNN_multi[i](x), y)

optimiser = ADAM(0.01);


loss_multi(i,x,y) = Flux.Losses.mse(model_CNN_multi[i](x), y)




#for loop questo
epochs = Int64
epochs = 200
loss_on_train = Array{Float64}(undef, numfiles, epochs)
loss_on_test = Array{Float64}(undef, numfiles, epochs)
best_params_CNN_multi = Vector{Any}(undef, numfiles)

loss_on_train[1,2]


for i in 1:numfiles 
    for epoch in 1:100
        Flux.train!(Flux.Losses.mse(model_CNN_multi[i](x), y), params_CNN_multiple[i],train_data[i], optimiser) 
        end
    end
end


for i in 1:numfiles 
    println("user ",i, " has started the training")
    for epoch in 1:2
        println("enter in epoch")
       # Flux.train!(loss_multi, params_CNN_multiple[i], train_data[i], optimiser)
        
    end
end
epochs = 
v_epochs = Int64[]
for epoch in 1:200
    push!(v_epochs, epoch)
end


for i in 1:numfiles 
    println("user ",i, " has started the training")
    
    for epoch in 1:200
        println("enter in epoch")
        Flux.train!((x, y) -> Flux.Losses.mse(model_CNN_multi[i](x), y), params_CNN_multiple[i], train_data[i], optimiser)
        println("training")
        #push!(epochs, epoch)
        #loss_on_train[i][epoch], loss_multi(i,x_train[i], y_train[i]))
       # push!(loss_on_test[i][epoch],  loss_multi(i,x_test[i], y_test[i]))

        @show epoch
        @show loss_multi(i,x_train[i], y_train[i])
        @show loss_multi(i,x_test[i], y_test[i])

        if epoch > 1
            if is_best(loss_on_test[i,epoch-1], loss_on_test[i,epoch])
                best_params_CNN_multi[i] = params_CNN_multiple[i]
            end
        end
     end
    println("user ",i, "has finished the training")
    #i = i+1
    println("user ",i, "will start the training")
end

for i in 1:numfiles
# Extract and add new trained parameters
    if isempty(best_params_CNN_multi[i])
        best_params_CNN_multi[i] = params_CNN_multiple[i]
    end

    Flux.loadparams!(model_CNN[i], best_params_CNN_multi[i]);
end


    plot(v_epochs, loss_on_train[20,:], lab="Training", c=:blue, lw=2);
    plot!(v_epochs, loss_on_test[20,:], lab="Test", c=:red, lw=2);
    title!("Recurrent architecture");
    yaxis!("Loss");
    xaxis!("Training epoch");
    savefig("r");

savefig("recurrent_loss");

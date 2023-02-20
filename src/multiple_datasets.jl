


numfiles = 254
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

for i in 1:numfiles
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

maxi = Float64[]
max_st = Float64[]
mini = Float64[]
min_st = Float64[]

for i in 1:numfiles
    push!(maxi, maximum(df[i]))
    push!(mini, minimum(df[i]))
    push!(max_st, maximum(df_st[i]))
    push!(min_st, minimum(df_st[i]))
end


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
    x_test[i] = df_st[i][train_range,:,:]
    y_test[i] = df_st[i][test_range,:,:]
    mean_df[i] = mean(df_new[i])
    st_df[i] = std(df_new[i])
    println("test of ",i," done")
end



ŷ_PM_multi_st= Vector{Any}(undef, numfiles)
ŷ_CNN_multi_st = Vector{Any}(undef, numfiles)
y_multi_st = Vector{Any}(undef, numfiles)
ŷ_CNN2_multi_st = Vector{Any}(undef, numfiles)

for i in 1:numfiles
    ŷ_PM_multi_st[i] = (model_PM_adam(x_test[i])[end-671:end])
    ŷ_CNN_multi_st[i] = (model_CNN_adam(x_test[i])[end-671:end])
    y_multi_st[i] = y_test[i][end-671:end]
    ŷ_CNN2_multi_st = (model_CNN_param_adj(x_test[i])[end-671:end])
end

ŷ_PM_multi_lbfgs_st = Vector{Any}(undef, numfiles)
ŷ_CNN_multi_lbfgs_st = Vector{Any}(undef, numfiles)
y_multi_lbfgs_st = Vector{Any}(undef, numfiles)
ŷ_CNN2_multi_lbfgs_st = Vector{Any}(undef, numfiles)









## plot
ŷ_PM_multi = Vector{Any}(undef, numfiles)
ŷ_CNN_multi = Vector{Any}(undef, numfiles)
y_multi = Vector{Any}(undef, numfiles)
ŷ_CNN2_multi = Vector{Any}(undef, numfiles)

for i in 1:numfiles
    ŷ_PM_multi[i] = ((model_PM_adam(x_test[i])[end-671:end]).*st_df[i]).+mean_df[i]
    ŷ_CNN_multi[i] = ((model_CNN_adam(x_test[i])[end-671:end]).*st_df[i]).+mean_df[i]
    y_multi[i] = (y_test[i][end-671:end].*st_df[i]).+mean_df[i]
    ŷ_CNN2_multi = ((model_CNN_param_adj(x_test[i])[end-671:end]).*st_df[i]).+mean_df[i]
end



p = Vector{Any}(undef, numfiles)
for i in 1:numfiles
    p[i] = plot( y_test[i] alpha = 0.4,  lab= y ,lw=2)
    plot!( ŷ_PM_multi[i],alpha = 0.4, lab= "ŷ PM", lw=2) 
    plot!( ŷ_CNN_multi[i], alpha = 0.4, lab= "ŷ CNN", lw=2)
    plot!( ŷ_CNN2_multi[i] , alpha = 0.4, lab= "ŷ CNN 2", lw=2)
    title!(string("Predicted vs true - ",user[i]));
    display(p[i])
    sleep(1)
    savefig(string(user[i],"_adam.pdf"))
end



ŷ_PM_multi_lbfgs = Vector{Any}(undef, numfiles)
ŷ_CNN_multi_lbfgs = Vector{Any}(undef, numfiles)
y_multi_lbfgs = Vector{Any}(undef, numfiles)
ŷ_CNN2_multi_lbfgs = Vector{Any}(undef, numfiles)


for i in 1:numfiles
    ŷ_PM_multi_lbfgs[i] = ((model_PM_lbfgs(x_test[i])[end-671:end]).*st_df[i]).+mean_df[i]
    ŷ_CNN_multi_lbfgs[i] = ((model_CNN_lbfgs(x_test[i])[end-671:end]).*st_df[i]).+mean_df[i]
    y_multi_lbfgs[i] = (y_test[i].*st_df[i]).+mean_df[i]
    ŷ_CNN2_multi_lbfgs[i] = ((model_CNN2_lbfgs(x_test[i])[end-671:end]).*st_df[i]).+mean_df[i]
end


p_lbfgs = Vector{Any}(undef, numfiles)
for i in 1:numfiles
    p_lbfgs[i] = plot( y_multi_lbfgs[i], alpha = 0.4,  lab= "y" ,lw=2)
    plot!( ŷ_PM_multi_lbfgs[i] ,alpha = 0.4, lab= "ŷ PM", lw=2) 
    plot!( ŷ_CNN_multi_lbfgs[i], alpha = 0.4, lab= "ŷ CNN", lw=2)
    plot!( ŷ_CNN2_multi_lbfgs[i] , alpha = 0.4, lab= "ŷ CNN 2", lw=2)
    title!(string("Predicted vs true - ",user[i]));
    display(p[i])
    sleep(1)
    savefig(string(user[i],"_lbfgs.pdf"))
end

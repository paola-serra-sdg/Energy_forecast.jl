#WORKING IN PROGESS
#do not check thic code
#read all files
for i in 1:numfiles-2
    tempdfs[i]= XLSX.readdata(files[i], " ACTIVA Y REACTIVA", "A4:G34272")   
end

#initialize vector 
dfs = Vector{Any}(undef, numfiles)
sheet = Vector{Any}(undef, numfiles)
s = Vector{String}(undef, numfiles)
p =  Vector{Any}(undef, numfiles)
X, Y = [], []

for i in 1:numfiles-2  #-2
    dfs[i]= XLSX.readxlsx(files[i])  
    sheet[i] = dfs[i][" ACTIVA Y REACTIVA"]
    s[i] = string(size(sheet[i][:],1))
    p[i] = sheet[i]["F4:F"*s[i]]
    p[i] = string.(p[i])
    p[i] = parse.(Float64, p[i])
    p[i] = Vector{Float64}(vec(p[i]))   
end


#Shift to one week
indice = Vector{Any}(undef, numfiles)
for i in 1:3
    indice[i] = enumerate(p[i])
end
X = Vector{Any}(undef, numfiles) 
Y = Vector{Any}(undef, numfiles)
for i in 1:numfiles-2
    indice[i] = enumerate(p[i])
end 



for i in 1:3
    indice[i] = enumerate(p[i])
    X[i] = []
    Y[i] = []
    
    for (ind,a) in indice[i]
        #X[i] = X[i][1:(size(p[i],1)-672+1)]
        #Y[i] = X[i][1:(size(p[i],1)-672+1)]
        if ind < size(p[i],1)-672+1
            a = p[i][ind]
            push!(X[i],a)
            push!(Y[i],p[i][ind + 672])
        end
        #end
    end

    X[i] = Float64.(X[i])
    Y[i]= Float64.(Y[i])
end

for i in 1:numfiles-2
    #println(size(df[i],1))
    println(string(i,":",size(X[i],1)))
end

for i in 1:70
    indi[i] = trunc(Int, 0.67*(size(df[i],1)))
    x_train[i] = X[i][1:indi[i]]
    y_train[i] = Y[i][indi[i]-671:indi[i]]
    x_test[i] = X[i][indi[i]+1:end]
    y_test[i] = Y[i][end-671:end]
end 


for i in 1:numfiles
train = st[1:end-672,:,:]   #train all obs less one week
test = st[673:end,:,:] 
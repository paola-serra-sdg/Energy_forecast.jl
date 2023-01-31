using Flux
using XLSX
using Flux.Data: DataLoader
using DataFrames
using Glob
using Plots
using Statistics: mean, std

#read all files in the directories
dirs = readdir("C:\\Users\\serrap\\Downloads\\Dati_Tesi") 
numfiles=length(dirs) 
files=glob("*.xlsx", ("C:\\Users\\serrap\\Downloads\\Dati_Tesi"))

#take only one dataset, in this case the first one
file = files[1]

#store only  active demand column
tempdf = XLSX.readxlsx(file)  
sheet = tempdf[" ACTIVA Y REACTIVA"]
s = string(size(sheet[:],1))
df = sheet["F4:F"*s]
timestamp = sheet["A4:A"*s]

#transform from Any in Float 
df = string.(df)
df = parse.(Float64, df)
#df = Vector{Float64}(vec(df))

#standardize the data 
st = std(df)
m = mean(df)
function standardize(data::Array)
    m = mean(data)
    s = std(data)
    st_x = (data.-m)./s
    return st_x
end


st = standardize(df)

#build x and y as a shift of n days of x
train = st[1:end-672,:,:]   #train all obs less one week
test = st[673:end,:,:]      #test all obs shifted of one week
#label for last week
y = df[end-671:end]


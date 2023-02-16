using Flux
using XLSX
using Flux.Data: DataLoader
using DataFrames
using Glob
using Plots
using Statistics: mean, std

#read all files in the directories
dirs = readdir("C:\\Users\\serrap\\Downloads\\Dati_Tesi") 
numfiles =length(dirs) 
files = glob("*.xlsx", ("C:\\Users\\serrap\\Downloads\\Dati_Tesi"))

#take only one dataset, in this case the first one
file = files[1]

#store only  active demand column
tempdf = XLSX.readxlsx(file)  

sheet = tempdf[" ACTIVA Y REACTIVA"]
s_single = string(size(sheet[:],1))
df_single = sheet["F4:F"*s_single]
timestamp = sheet["A4:A"*s_single]

#transform from Any in Float 
df_single = string.(df_single)
df_single = parse.(Float64, df_single)
#df = Vector{Float64}(vec(df))

#standardize the data 
function standardize(data::Array)
    m = mean(data)
    s = std(data)
    st_x = (data.-m)./s
    return st_x
end
m1 = mean(df_single)
s1 = std(df_single)
st = standardize(df_single)

#build x and y as a shift of n days of x
#splitting train and test
week = 43
week_length = 4 * 24 * 7
start_week_prediction = week * week_length  #44*672
end_week_prediction = start_week_prediction + week_length 
train_range = 1:start_week_prediction  # arriva asll'inizio 44 esima settimana
test_range= start_week_prediction+1 : end_week_prediction

672*52

#splitting in train and test

X_train = st[1:end-672,:,:]
Y_train = st[673:end,:,:]

X_test = X[end-671:end,:,:]
Y_test = Y[end-671:end,:,:]
# X = st[1:end-week_length,:,:]  
# Y = st[week_length+1:end,:,:]

# X_train = X[1:start_week_prediction,:,:]# inizio 43
# Y_train = Y[1:start_week_prediction,:,:]# 43 inclusa

# X_test = X[test_range,:,:] #43
# Y_test = Y[test_range,:,:] #44
#IL TEST



# X = st[train_range,:,:]
# Y = st[train_range,:,:]


# X_test = st[test_range,:,:]
# Y_test = st[test_range,:,:]

train_data_single = DataLoader((X_train, Y_train) ; batchsize = 1)
test_data_single = DataLoader((X_test, Y_test); batchsize = 1)


#Y ground truth for week 44
y_st = Y[test_range]
y = (y_st.*s1).+m1

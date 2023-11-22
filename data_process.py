import pandas as pd

file_names = ['BarclaysUSAggregateTreasury.xlsx','BarclaysUSCorporateHighYield.xlsx','BarclaysUSCorporateInvestmentGrade.xlsx',
              'BarclaysUSLongTreasury.xlsx','FTSEEPRANAREITDevelopedTotalReturnIndex.xlsx','LBMAGoldPrice.xlsx',
              'MSCIEmergingMarketIndex.xlsx','MSCIWorldIndex.xlsx','SP500TotalReturnIndex.xlsx','SPGSCIIndex.xlsx']

df = pd.DataFrame()
for i in range(len(file_names)):
    d = pd.read_excel(file_names[i],sheet_name='Worksheet')
    d = d.iloc[6:,:2]
    d.columns = ['Dates', file_names[i]]
    d.index = pd.to_datetime(d['Dates'])
    df[file_names[i]] = d[file_names[i]]
    print(df.shape)
print(df)
df.to_csv('index_data.csv')
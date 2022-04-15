# Data Processing Script
# Author: Renato Okabayashi Miyaji
# Date: 28-11-2021


"""
Read the data from the Atmospheric Radiation Measurement (ARM) repository collected by the G-1 aircraft of the GOAmazon 2014/15 project,
interpolate and generate a dataset for each variable.
Then, the bioclimatic dataset is generated through the manipulation of species occurrence datasets from the Global Information Biodiversity 
Facility (GBIF) and the Portal da Biodiversidade do Instituto Chico Mendes de Conservação da Biodiversidade (ICMBio) and the join operation 
between these datasets and the one resultant from the interpolation process.
"""


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from statistics import mean


# Execution parameters
list_variables = ['co(ppm)', 'o3(ppb)', 'nox(ppb)', 'CPC3010', 'Acetonitrile(ppb)', 'Isoprene(ppb)']
season = 'dry'
file_name = 'Interpolated_Data_{}'.format(season)
file_name_1 = 'portal_bio'
file_name_2 = 'GBIF'
ini_date = '2009-01-01'
end_date = '2019-01-01'
min_lat = -3.5
min_long = -60.8
max_lat = -2.8
max_long = -59.8


def read_files(season):
    """
    Read the files and filter by flight number
    """
    if season == 'dry':
        g1_d17 = pd.read_excel("G1_dryseason_17.xlsx")
        g1_d18 = pd.read_excel("G1_dryseason_18.xlsx")
        g1_d19 = pd.read_excel("G1_dryseason_19.xlsx")
        g1_d20 = pd.read_excel("G1_dryseason_20.xlsx")
        g1_d21 = pd.read_excel("G1_dryseason_21.xlsx")
        g1_d22 = pd.read_excel("G1_dryseason_22.xlsx")
        g1_d23 = pd.read_excel("G1_dryseason_23.xlsx")
        g1_d24 = pd.read_excel("G1_dryseason_24.xlsx")
        g1_d25 = pd.read_excel("G1_dryseason_25.xlsx")
        g1_d26 = pd.read_excel("G1_dryseason_26.xlsx")
        g1_d27 = pd.read_excel("G1_dryseason_27.xlsx")
        g1_d28 = pd.read_excel("G1_dryseason_28.xlsx")
        g1_d29 = pd.read_excel("G1_dryseason_29.xlsx")
        g1_d30 = pd.read_excel("G1_dryseason_30.xlsx")
        g1_d31 = pd.read_excel("G1_dryseason_31.xlsx")
        g1_d32 = pd.read_excel("G1_dryseason_32.xlsx")
        g1_d33 = pd.read_excel("G1_dryseason_33.xlsx")
        
        flights = [g1_d17,g1_d18,g1_d19,g1_d20,g1_d21,g1_d22,g1_d23,g1_d24,g1_d25,g1_d26,g1_d27,g1_d28,g1_d29,g1_d30,g1_d31,g1_d32,g1_d33]
        return flights
    
    elif season == 'wet':
        g1_d1 = pd.read_excel("G1_wetseason_1.xlsx")
        g1_d2 = pd.read_excel("G1_wetseason_2.xlsx")
        g1_d3 = pd.read_excel("G1_wetseason_3.xlsx")
        g1_d4 = pd.read_excel("G1_wetseason_4.xlsx")
        g1_d5 = pd.read_excel("G1_wetseason_5.xlsx")
        g1_d6 = pd.read_excel("G1_wetseason_6.xlsx")
        g1_d7 = pd.read_excel("G1_wetseason_7.xlsx")
        g1_d8 = pd.read_excel("G1_wetseason_8.xlsx")
        g1_d9 = pd.read_excel("G1_wetseason_9.xlsx")
        g1_d10 = pd.read_excel("G1_wetseason_10.xlsx")
        g1_d11 = pd.read_excel("G1_wetseason_11.xlsx")
        g1_d12 = pd.read_excel("G1_wetseason_12.xlsx")
        g1_d13 = pd.read_excel("G1_wetseason_13.xlsx")
        g1_d14 = pd.read_excel("G1_wetseason_14.xlsx")
        g1_d15 = pd.read_excel("G1_wetseason_15.xlsx")
        g1_d16 = pd.read_excel("G1_wetseason_16.xlsx")
        
        flights = [g1_d1,g1_d2,g1_d3,g1_d4,g1_d5,g1_d6,g1_d7,g1_d8,g1_d9,g1_d10,g1_d11,g1_d12,g1_d13,g1_d14,g1_d15,g1_d16]
        return flights

    
def select_colums(df_flight, variable):
    """
    Select only the relevant columns of the dataframe
    """
    
    df_flight = df_flight.drop(columns=['Unnamed: 0'])
    df_flight = df_flight[['lat','long','alt',variable]]
    df_flight = df_flight.dropna(subset=[variable])
    
    return df_flight


def filter_alt(df_flight):
    """
    Filter the dataframe by altitude
    """
    
    max_alt = 1500;
    df_flight = df_flight[df_flight['alt']<=max_alt].reset_index(drop=True)
    
    return df_flight


def maxmin_season(season, flights):
    """
    Calculate the maximum and minimum from latitude and logitude of each season 
    """

    l_max_lat = []
    l_max_lon = []
    l_min_lat = []
    l_min_lon = []

    for flight in flights:
        l_max_lat.append(flight['lat'].max())
        l_min_lat.append(flight['lat'].min())
        l_max_lon.append(flight['long'].max())
        l_min_lon.append(flight['long'].min())
    max_lat = max(l_max_lat)
    max_lon = max(l_max_lon)
    min_lat = min(l_min_lat)
    min_lon = min(l_min_lon)

    return max_lat, max_lon, min_lat, min_lon


def dxdy(df_flight):
    """
    Calculate the discretization of latitude and longitude
    """
    
    dxs = []
    dys = []

    for i in range(len(df_flight)-1):
        dlat = abs(df_flight['lat'][i]-df_flight['lat'][i+1])
        dys.append(dlat)
        dlong = abs(df_flight['long'][i]-df_flight['long'][i+1])
        dxs.append(dlong)
    
    return max(dys), max(dxs)


def dxdy_season(season, flights):
    """
    Calculate the discretization of latitude and longitude for the season
    """   
    
    dxs = []
    dys = []

    for flight in flights:
        dxs.append(dxdy(flight)[0])
        dys.append(dxdy(flight)[1])
    dx = min(dxs)
    dy = min(dys)
        
    return dx,dy


def interpolate(df_flight,variable,max_lat,max_lon,min_lat,min_lon,dx,dy):
    """
    Interpolate the variable under the specified conditions
    """
    
    # Coordinates and variable
    x = df_flight['long']
    y = df_flight['lat']
    z = df_flight[variable]

    # Create the mesh
    xi = np.arange(min_lon,max_lon,dx) 
    yi = np.arange(min_lat,max_lat,dy)
    xi,yi = np.meshgrid(xi,yi)
    xi_ = np.arange(min_lon,max_lon,dx) 
    yi_ = np.arange(min_lat,max_lat,dy)

    # Interpolate
    zi = griddata((x,y),z,(xi,yi),method='linear')
    
    return zi,xi_,yi_,x,y,dx,dy


def plot_interpolation(variable,xi,yi,zi,x,y):
    """
    Plot the result of the interpolation
    """
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    plt.contourf(xi,yi,zi)
    plt.plot(x,y,'k.')
    plt.xlabel('Longitude (°)')
    plt.ylabel('Latitude (°)')
    plt.title('Intepolation of {}'.format(variable))
    plt.colorbar()
    

def create_dataframe(min_lon,max_lon,dx,min_lat,max_lat,dy,zi):
    """
    Create the dataframe resulting of the interpolation 
    """
    
    xi = np.arange(min_lon,max_lon,dx) 
    yi = np.arange(min_lat,max_lat,dy)
    
    matriz = []
    for i in range(len(yi)):
        linha = []
        for j in range(len(xi)):
            linha.append(zi[i][j])
        matriz.append(linha) 
    interp = pd.DataFrame(matriz)
    interp.index = yi
    interp.columns = xi
    
    return interp


def interpolate_variable(df_flight, variable, season, flights):
    """
    Interpolate the variable for the selected season
    """
    
    # Select columns
    df_flight = select_colums(df_flight, variable)
    
    if not df_flight.empty:
        # Filter the altitude
        f_df_flight = filter_alt(df_flight)
        
        if f_df_flight.shape[0] > 100:

            # Calculate the maximum and minimum of latitude and longitude
            max_lat, max_lon, min_lat, min_lon = maxmin_season(season, flights)

            # Calculate the discretization
            dx,dy = dxdy_season(season, flights)

            # Interpolate
            zi,xi,yi,x,y,dx,dy = interpolate(f_df_flight,variable,max_lat,max_lon,min_lat,min_lon,dx,dy)

            # Plot the interpolation
            #plot_interpolation(variable,xi,yi,zi,x,y)

            # Create dataframe
            interp = create_dataframe(min_lon,max_lon,dx,min_lat,max_lat,dy,zi)

            return interp
   
    
def summarization(min_lon,max_lon,dx,min_lat,max_lat,dy,list_flights):
    """
    Summarize the interpolations of the variable in the season
    """
    
    xi = np.arange(min_lon,max_lon,dx) 
    yi = np.arange(min_lat,max_lat,dy)
    
    z_final = []
    
    # Threshold 
    minimum = len(list_flights)/2;
    
    for l in range(len(list_flights)):
        if type(list_flights[l]) == pd.core.frame.DataFrame:
            flight_df = list_flights[l]
            
    for i in range(len(flight_df)):
        line = []
        for j in range(len(flight_df.columns)):
            counter = 0
            lista = []
            for k in range(len(list_flights)):
                flight = list_flights[k]
                if type(flight) == pd.core.frame.DataFrame:
                    if np.isnan(flight.iloc[i,j]) == False:
                        lista.append(flight.iloc[i,j])
                        counter += 1
            if len(lista) > minimum:
                idmax = lista.index(max(lista))
                lista[idmax] = 0
                line.append(mean(lista))
            else:
                line.append(None)
        z_final.append(line)
    
    z_final = pd.DataFrame(z_final)
    z_final.index = yi
    z_final.columns = xi

    return xi,yi,z_final


def plot_summarization(variable,xi,yi,z_final):
    """
    Plot the summarization 
    """
    
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    plt.contourf(xi,yi,z_final)
    plt.xlabel('Longitude (°)')
    plt.ylabel('Latitude (°)')
    plt.title('Interpolation of {}'.format(variable))
    plt.colorbar()
    

def interpolation_season(variable, season):
    """
    Interpolate the variable during the season
    """
    
    # Read the files
    if season == 'dry':
        flights = read_files(season) 
    elif season == 'wet':
        flights = read_files(season)
    
    # Calculate the maximum and minimum of latitude and longitude
    max_lat, max_lon, min_lat, min_lon = maxmin_season(season, flights)

    # Calculate the discretization 
    dx,dy = dxdy_season(season, flights)
    
    lista_flights = []
    for flight in flights:
        interpolation = interpolate_variable(flight, variable, season, flights)
        lista_flights.append(interpolation)

    # Summarize the interpolations
    xi,yi,z_final = summarization(min_lon,max_lon,dx,min_lat,max_lat,dy,lista_flights)

    # Plot 2D
    #plot_summarization(variable,xi,yi,z_final)
    
    return z_final


def generate_dataset(season, list_variables, file_name):
    """
    Generates the interpolation for each variable from the list and a dataframe
    """
    
    # Iterates and generates the interpolation for each variable
    for i in range(len(list_variables)):
        if i == 0:
            df_season = interpolation_season(list_variables[i], season)
        
            # Stack the dataset
            df_season = pd.DataFrame(df_season.stack(dropna=False)).reset_index()
            df_season = df_season.rename(columns={'level_0': 'latitude', 'level_1': 'longitude', 0: list_variables[i]})
        
        else:
            df_interpolated = interpolation_season(list_variables[i], season)
        
            # Stack the dataset
            df_interpolated = pd.DataFrame(df_interpolated.stack(dropna=False)).reset_index()
            df_interpolated = df_interpolated.rename(columns={'level_0': 'latitude', 'level_1': 'longitude', 0: list_variables[i]})
        
            # Merge
            df_season = df_season.merge(df_interpolated, on=['latitude', 'longitude'])
            
    # Filter for not NaN
    df_season = df_season.dropna(subset=list_variables, how='all').reset_index(drop=True)
    
    # Export data
    df_season.to_excel('{}.xlsx'.format(file_name), index=False)
    
    return df_season 


def read_bio_files(file_name):
    """
    Read the datasets with species occurrence data
    """
    
    if file_name == 'portal_bio':
        
        df = pd.read_excel('portalbio_export_16-08-2020-20-20-34.xlsx')
        
        # Select columns
        df = df[['Data do registro','Nome cientifico','Nome comum','Nivel taxonomico', 'Numero de individuos', 'Reino', 'Filo', 'Classe','Ordem', 'Familia', 'Genero', 'Especie', 'Localidade', 'Pais', 'Estado/Provincia','Municipio', 'Latitude', 'Longitude']]
        df = df.rename(columns={'Data do registro':'date', 'Latitude':'latitude', 'Longitude':'longitude', 'Especie':'species'})
        df = df[['date','latitude','longitude','species']]
        
        # Round coordinates
        df['latitude'] = df['latitude'].apply(lambda x: x/(10**(len(str(x))-2)))
        df['longitude'] = df['longitude'].apply(lambda x: x/(10**(len(str(x))-3)))
        
        # Filter unknown species
        df = df[df['species'] != 'Sem Informações'].reset_index(drop = True)
        
        return df
    
    elif file_name == 'GBIF':
        
        df = pd.read_excel('GBIF_occu.xlsx')
        
        # Select columns
        df = df[['eventDate','stateProvince','county','municipality','locality','decimalLatitude','decimalLongitude','kingdom','phylum','class','order','family','genus','species']]
        df = df.rename(columns={'eventDate':'date', 'decimalLatitude':'latitude', 'decimalLongitude':'longitude'})
        df = df[['date','latitude','longitude','species']]
        
        # Round coordinates
        df['latitude'] = df['latitude'].apply(lambda x: x/(10**(len(str(x))-4)))
        df['longitude'] = df['longitude'].apply(lambda x: x/(10**(len(str(x))-5)))
        
        # Filter unknown species        
        df = df.dropna(subset = ['species']).reset_index(drop = True)
        
        return df
    

def filter_files(df,ini_date, end_date, min_lat, min_long, max_lat, max_long):
    """
    Filter the datasets by date and coordinates (latitude and longitude)
    """
    
    # Filter by date
    df = df[df['date']>=ini_date]
    df = df[df['date']<=end_date]
    df = df.sort_values(by='date').reset_index(drop = True)
    
    # Filter by coordinates
    df = df[df['latitude'] >= min_lat]
    df = df[df['latitude'] <= max_lat]
    df = df[df['longitude'] >= min_long]
    df = df[df['longitude'] <= max_long]
    df = df.reset_index(drop = True)
    
    return df


def concatenate_dfs(df1, df2):
    """
    Concatenate the species occurrence datasets
    """

    df = pd.concat([df1,df2])
    df = df.sort_values(by='date').reset_index(drop = True)
    
    # Drop duplicates
    df = df.drop_duplicates(subset=['latitude', 'longitude', 'date', 'species'], keep='first').reset_index(drop=True)
    
    return df


def select_season(season, df):
    """
    Filter the dataset by season
    """
    
    if season == 'dry':
        
        # Filter by season
        mask1 = df['date'].map(lambda x: x.month) == 8
        mask2 = df['date'].map(lambda x: x.month) == 9
        mask3 = df['date'].map(lambda x: x.month) == 10
        df_1 = df[mask1]
        df_2 = df[mask2]
        df_3 = df[mask3]
        df_dry = pd.concat([df_1,df_2,df_3], ignore_index=True)
        
        return df_dry
        
    elif season == 'wet':
        
        # Filter by season
        mask1 = df['date'].map(lambda x: x.month) == 1
        mask2 = df['date'].map(lambda x: x.month) == 2
        mask3 = df['date'].map(lambda x: x.month) == 3
        df_1 = df[mask1]
        df_2 = df[mask2]
        df_3 = df[mask3]
        df_wet = pd.concat([df_1,df_2,df_3], ignore_index=True)
        
        return df_wet
    

def merge_datasets(season, df):
    """
    Merge the datasets in order to generate the bioclimatic dataset
    """
    
    # Read interpolated data
    df_interpolated = pd.read_excel('Interpolated_Data_{}.xlsx'.format(season))
    
    # Format coordinates
    df_interpolated['latitude_'] = df_interpolated['latitude'].apply(lambda x: round(x,3))
    df_interpolated['longitude_'] = df_interpolated['longitude'].apply(lambda x: round(x,3))
    df['latitude_'] = df['latitude'].apply(lambda x: round(x,3))
    df['longitude_'] = df['longitude'].apply(lambda x: round(x,3))
    
    # Join
    df = df.merge(df_interpolated, on=['latitude_', 'longitude_']).reset_index(drop=True)
    
    # Count the numbers of occurrences
    occurrences = pd.DataFrame(df.groupby(by=['species'])['date'].count())
    occurrences = occurrences.rename(columns={'date':'qty'})
    
    # Filter minimum quantity
    occurrences = occurrences[occurrences['qty'] >= 17].reset_index()
    
    # Select columns
    list_columns = list(df.columns)
    list_columns.remove('latitude_x')
    list_columns.remove('longitude_x')
    list_columns.remove('latitude_')
    list_columns.remove('longitude_')
    df = df[list_columns]
    df = df.rename(columns={'latitude_y':'latitude','longitude_y':'longitude'})
    
    # Create new column
    df_interpolated.loc[:,'species'] = None

    # Add key column
    def key(lat,lon):
        return str(lat)+str(lon)

    df_interpolated['key'] = df_interpolated.apply(lambda x: key(x['latitude'],x['longitude']), axis=1)
    df['key'] = df.apply(lambda x: key(x['latitude'],x['longitude']), axis=1)

    # Filtering
    df_interpolated = df_interpolated.loc[~df_interpolated['key'].isin(df['key'].to_list())]
    df_interpolated = pd.concat([df_interpolated,df]).reset_index(drop=True)
    l_columns = occurrences['species'].to_list()+[None]
    df_interpolated = df_interpolated.loc[df_interpolated['species'].isin(l_columns)].reset_index(drop=True)
    
    # Encoding
    df_enc = pd.get_dummies(df_interpolated.species, prefix='species')

    # Add column
    df_bioclim = pd.merge(df_interpolated, df_enc, left_index=True, right_index=True)
    
    # Drop NaN
    list_columns.remove('latitude_y')
    list_columns.remove('longitude_y')
    list_columns.remove('date')
    list_columns.remove('species')
    df_bioclim = df_bioclim.dropna(subset=list_columns)
    
    # Drop columns
    df_bioclim.drop(labels=['latitude_','longitude_','species','key','date'], axis=1)
    df_bioclim = df_bioclim.reset_index(drop=True)
    
    df_bioclim.to_csv('Bioclimatic_dataset_{}.csv'.format(season), index=False)
    
    return df_bioclim

 
def main():
    
    # Perform the spatial interpolation
    dataset = generate_dataset(season, list_variables, file_name)
    
    # Generate the bioclimatic dataset
    df1 = read_bio_files(file_name_1)
    df2 = read_bio_files(file_name_2)
    df1 = filter_files(df1,ini_date, end_date, min_lat, min_long, max_lat, max_long)
    df2 = filter_files(df2,ini_date, end_date, min_lat, min_long, max_lat, max_long)
    df = concatenate_dfs(df1, df2)
    df = select_season(season, df)
    df_bioclim = merge_datasets(season, df)


main()
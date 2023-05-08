'''Q2'''
#import relevent libraries
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import IsolationForest


'''Q2 - Section a'''
#This function gets two columns relevant to create the graph
#from the relevant df and issues the graph out
def create_plot(x_column, y_column):
    sns.set()
    warnings.filterwarnings('ignore')
    sns.lineplot(x=x_column, y=y_column)
    plt.title('Distance of detected object as a function of time')
    plt.xlabel('Time (sec)')
    plt.ylabel('Dsitance (meters)')
    plt.tight_layout()
    plt.show()

'''Q2 - Section b'''
#This function receives df and the name of the relevant column in which
#we are looking for the exceptional values, the function returns the df
#after checking each value of a cell within the selected column from the df,
#if it meets the conditions the record goes up to the new df
def is_kinematic_point_z_valid(df, y_column_name):
    new_df = df.loc[df[y_column_name]>=0]
    return new_df

'''Q2 - Section c'''
#This function receives df and the names of the three columns with which we
#perform the calculation of the total average velocity according to the average of the
#instantaneous velocity and at the same time the calculation of the total constant velocity in df          
def calc_velocity(df, x_column_name, y_column_name, z_column_name):
    #calculation of the total constant velocity
    apx_con_vel = (df.loc[df.last_valid_index()][y_column_name] - df.loc[df.first_valid_index()][y_column_name]) \
        /(df.loc[df.last_valid_index()][x_column_name] - df.loc[df.first_valid_index()][x_column_name])
    
    df = df.reset_index()
    df[z_column_name]=df[z_column_name].astype(float)
    #calculate the instantaneous velocity for each index and enter the value in the relevant column in df
    for index in df.index:
        try:
            df.at[index, z_column_name] = (df.loc[index+1][y_column_name]-df.loc[index][y_column_name])\
                /(df.loc[index+1][x_column_name]-df.loc[index][x_column_name])
        except:
            df.at[index, z_column_name] = df.at[index-1, z_column_name]
    #calculating the average of all momentary velocities        
    apx_con_avg_vel = df[z_column_name].mean()
    return apx_con_vel, apx_con_avg_vel 

'''Q2 - Bonus Q'''
#This function gets df and two names of columns on the basis of which we want 
#to teach the model,
#The model chosen for finding the outliers data is IsolationForest,
#assuming that the distribution of existing data is not symmetrically normal,
#We also assume that an exceptional figure is constructed from the distance
#value as a function of time and therefore an outliers model should be constructed as clusters
def remove_outliers(df, column_name_x, column_name_y):
    iso_model = IsolationForest(max_samples=100, random_state=1, contamination=0.03)
    outliers_preds = iso_model.fit_predict(detection_valid_df[[column_name_x, column_name_y]])
    filtered_dataframe = detection_valid_df.loc[outliers_preds==1]
    return filtered_dataframe
   

if __name__ == '__main__':
    
    '''Q2 - Section a'''
    #read file:
    detection_df = pd.read_csv('Q2_detection_file_2.csv')
    #create a graph
    create_plot(detection_df['time_sec'], detection_df['rw_kinematic_point_z'])
    
    '''Q2 - Section b'''
    #clean_df:
    new_detection_df = is_kinematic_point_z_valid(detection_df, 'rw_kinematic_point_z')
    #create a graph
    create_plot(new_detection_df['time_sec'], new_detection_df['rw_kinematic_point_z'])
    
    
    '''Q2 - Section c'''
    # calc the average of all momentary velocities & total constant velocity
    calc_ap_cons_vel = calc_velocity(new_detection_df, 'time_sec', 'rw_kinematic_point_z', 'velocity_z')
    print(f"constant velocity = {calc_ap_cons_vel[0]} , averaged constant velocity = {calc_ap_cons_vel[1]}")
    
    '''Q2 - Bonus Q'''
    #read file:
    detection_valid_df = pd.read_csv('Q2_detection_file_2_bonus.csv')
    #remove outliers
    detection_valid_clean_df = remove_outliers(detection_valid_df, 'time_sec', 'rw_kinematic_point_z')
    #create a graph
    create_plot(detection_valid_clean_df['time_sec'], detection_valid_clean_df['rw_kinematic_point_z'])
    

    




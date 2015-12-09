#!/usr/bin/env python
""" Calibration fit for eQuest

Creates weather normalized utility bills
Loads monthly values from eQuest for any number of meters
Plot the calibrations and table of CV(RMSE) and NMBE
"""
# Make it python 2.x and 3.x compatible
from __future__ import division, print_function


#Import modules
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.dates as md
import matplotlib.pyplot as plt
plt.style.use('ggplot')
#import seaborn as sns
#sns.set(style="white", context="talk")


import glob as gb
import os
import sys

import re

import datetime as dt

    
    
def create_pv_a_dict():
    """
    Initializes a dictionary of dataframes for the PV-A report
    
    Args: None
    ------ 
        
    Returns:
    --------
        pv_a_dict(dict of pd.DataFrame): a dictionary of dataframes to collect
            the PV-A reports
            
    Needs:
    -------------------------------
        import pandas as pd
    
    """
    
    pv_a_dict = {}

    ###### CIRCULATION LOOPS 
    loop_string = 'CIRCULATION LOOPS'
    loop_info_cols = ['Heating Cap. (mmBTU/hr)',
                     'Cooling Cap. (mmBTU/hr)',
                     'Loop Flow (GPM)',
                     'Total Head (ft)',
                     'Supply UA (BTU/h.F)',
                     'Supply Loss DT (F)',
                     'Return UA (BTU/h.F)',
                     'Return Loss DT (F)',
                     'Loop Volume (gal)',
                     'Fluid Heat Cap. (BTU/lb.F)']
    df = pd.DataFrame(columns=loop_info_cols)
    df.index.name = 'Circulation Loop'
    pv_a_dict[loop_string] = df

    ###### PUMPS
    pump_string = 'PUMPS'
    pump_info_cols = ['Attached to',
                     'Flow (GPM)',
                     'Head (ft)',
                     'Head Setpoint (ft)',
                     'Capacity Control',
                     'Power (kW)',
                     'Mech. Eff',
                     'Motor Eff']
    df = pd.DataFrame(columns=pump_info_cols)
    df.index.name = 'Pump'
    pv_a_dict[pump_string] = df

    ###### PRIMARY EQUIPMENT (Chillers, boilers)
    primary_string = 'PRIMARY EQUIPMENT'
    primary_info_cols = ['Equipment Type',
                         'Attached to',
                         'Capacity (mmBTU/hr)',
                         'Flow (GPM)',
                         'EIR',
                         'HIR',
                         'Aux. (kW)']
    df = pd.DataFrame(columns=primary_info_cols)
    df.index.name = 'Primary Equipment'
    pv_a_dict[primary_string] = df

    ###### COOLING TOWERS
    ct_string = 'COOLING TOWERS'
    ct_info_cols = ['Equipment Type',
                     'Attached to',
                     'Cap. (mmBTU/hr)',
                     'Flow (GPM)',
                     'Nb of Cells',
                     'Fan Power per Cell (kW)',
                     'Spray Power per Cell (kW)',
                     'Aux. (kW)']
    df = pd.DataFrame(columns=ct_info_cols)
    df.index.name = 'Cooling Tower'
    pv_a_dict[ct_string] = df

    ###### DHW Heaters
    dhw_string = 'DW-HEATERS'
    dhw_info_cols = ['Equipment Type',
                     'Attached to',
                     'Cap. (mmBTU/hr)',
                     'Flow (GPM)',
                     'EIR',
                     'HIR',
                     'Auxiliary (kW)',
                     'Tank (Gal)',
                     'Tank UA (BTU/h.ft)']
    df = pd.DataFrame(columns=dhw_info_cols)
    df.index.name = 'DHW Heaters'
    pv_a_dict[dhw_string] = df
    
    return pv_a_dict
    
 
def post_process_pv_a(pv_a_dict, output_to_csv=True):
    """
    Convert the dataframes in the dictionary to numeric dtype 
    and calculates some efficiency metrics, such as Chiller COP, Pump kW/GPM, etc.
    
    Args:
    ------
        pv_a_dict(dict of pd.DataFrame): dictionary of dataframes
            that has the PV-A info
        
        output_to_csv (boolean): whether you want to output 'PV-A.csv'
        
    Returns:
    --------
        pv_a_dict(dict of pd.DataFrame): dataframes in numeric dtype and more metrics
        
        Also spits out a 'PV-A.csv' file if required.
            
        
    Needs:
    -------------------------------
        import pandas as pd
    
    """
    
    # Convert numeric for circulation loops
    df_circ = pv_a_dict['CIRCULATION LOOPS']
    df_circ = df_circ.apply(lambda x: pd.to_numeric(x))

    # Calculate kW/GPM for pumps
    df_pumps = pv_a_dict['PUMPS']
    num_cols = ['Flow (GPM)', 'Head (ft)', 'Head Setpoint (ft)', 'Power (kW)', 'Mech. Eff', 'Motor Eff']
    df_pumps[num_cols] = df_pumps[num_cols].apply(lambda x: pd.to_numeric(x))
    df_pumps['W/GPM'] = 1000*df_pumps['Power (kW)'] / df_pumps['Flow (GPM)']

    # Calculate fan kW/GPM for cooling towers
    df_ct = pv_a_dict['COOLING TOWERS']
    num_cols = ['Cap. (mmBTU/hr)', 'Flow (GPM)', 'Nb of Cells', 'Fan Power per Cell (kW)', 'Spray Power per Cell (kW)', 'Aux. (kW)']
    df_ct[num_cols] = df_ct[num_cols].apply(lambda x: pd.to_numeric(x))
    df_ct['Fan W/GPM'] = 1000*df_ct['Fan Power per Cell (kW)'] * df_ct['Nb of Cells'] / df_ct['Flow (GPM)']
    # GPM per ton
    df_ct['GPM/ton'] =  df_ct['Flow (GPM)'] * 12 /(1000*df_ct['Cap. (mmBTU/hr)'])


    # Calculate proper efficiency for primary equipment    
    # First, convert to numeric
    df_primary = pv_a_dict['PRIMARY EQUIPMENT']
    num_cols = [ 'Capacity (mmBTU/hr)', 'Flow (GPM)', 'EIR', 'HIR', 'Aux. (kW)']
    df_primary[num_cols] = df_primary[num_cols].apply(lambda x: pd.to_numeric(x))

    # Separate between chillers and boilers
    boilers = df_primary['Equipment Type'].str.contains('HW')
    df_boilers = df_primary.ix[boilers].copy()
    df_chillers = df_primary.ix[~boilers].copy()
    # Delete from dict
    del pv_a_dict['PRIMARY EQUIPMENT']
    
    # Deal with boilers first
    df_boilers['Thermal Eff'] = 1 / df_boilers['HIR']
    # Assign that to the pv_a_dict
    pv_a_dict['BOILERS'] = df_boilers

    # Chillers
    df_chillers['COP'] = 1/  df_chillers['EIR']
    # KW/ton = 12 / (COP x 3.412)
    df_chillers['kW/ton'] = 12 / (df_chillers['COP'] * 3.412)
    # GPM/ton
    df_chillers['GPM/ton'] =  df_chillers['Flow (GPM)'] * 12 /(1000*df_chillers['Capacity (mmBTU/hr)'])
    
    pv_a_dict['CHILLERS'] = df_chillers

    # DW-HEATERs
    df_dhw = pv_a_dict['DW-HEATERS']
    num_cols = ['Cap. (mmBTU/hr)', 'Flow (GPM)', 'EIR', 'HIR', 'Auxiliary (kW)', 'Tank (Gal)', 'Tank UA (BTU/h.ft)']
    df_dhw[num_cols] = df_dhw[num_cols].apply(lambda x: pd.to_numeric(x))
    df_dhw['Thermal Eff'] = 1 / df_dhw['HIR'] 
    
    
    # Output to CSV
    if sys.version_info < (3,0):
        with open('PV-A.csv', 'wb') as f:
            print('PV-A Report\n\n', file=f)
        with open('PV-A.csv', 'ab') as f:
            for k, v in pv_a_dict.items():
                print(k, file=f)
                v.to_csv(f)
                print('',file=f)
    else:
        with open('PV-A.csv', 'w') as f:
            print('PV-A Report\n\n', file=f)
        with open('PV-A.csv', 'a') as f:
            for k, v in pv_a_dict.items():
                print(k, file=f)
                v.to_csv(f)
                print('',file=f)
    
    return pv_a_dict
    
    
def create_sv_a_dict():
    """
    Initializes a dictionary of dataframes for the SV-A report
    
    Args: None
    ------ 
        
    Returns:
    --------
        sv_a_dict(dict of pd.DataFrame): a dictionary of dataframes to collect
            the SV-A reports
            Has three keys: 'Systems', 'Fans', 'Zones'
            
    Needs:
    -------------------------------
        import pandas as pd
    
    """
    
    sv_a_dict = {}
    
    system_info_cols = ['System Type',
                        'Altitude Factor',
                        'Floor Area (sqft)',
                        'Max People',
                        'Outside Air Ratio',
                        'Cooling Capacity (kBTU/hr)',
                        'Sensible (SHR)',
                        'Heating Capacity (kBTU/hr)',
                        'Cooling EIR (BTU/BTU)',
                        'Heating EIR (BTU/BTU)',
                        'Heat Pump Supplemental Heat (kBTU/hr)']
                        
    system_info = pd.DataFrame(columns=system_info_cols)
    system_info.index.name = 'System'
    sv_a_dict['Systems'] = system_info

    fan_info_cols = ['Capacity (CFM)',
                     'Diversity Factor (FRAC)',
                     'Power Demand (kW)',
                     'Fan deltaT (F)',
                     'Static Pressure (in w.c.)',
                     'Total efficiency',
                     'Mechanical Efficiency',
                     'Fan Placement',
                     'Fan Control',
                     'Max Fan Ratio (Frac)',
                     'Min Fan Ratio (Frac)']
    index = pd.MultiIndex(levels=[['System'],['Fan Type']],
                          labels=[[],[]],
                          names=[u'System', u'Fan Type'])
    fan_info = pd.DataFrame(index=index, columns=fan_info_cols)
    sv_a_dict['Fans'] = fan_info


    zone_info_cols = ['Supply Flow (CFM)',
                     'Exhaust Flow (CFM)',
                     'Fan (kW)',
                     'Minimum Flow (Frac)',
                     'Outside Air Flow (CFM)',
                     'Cooling Capacity (kBTU/hr)',
                     'Sensible (FRAC)',
                     'Extract Rate (kBTU/hr)',
                     'Heating Capacity (kBTU/hr)',
                     'Addition Rate (kBTU/hr)',
                     'Zone Mult']
    index = pd.MultiIndex(levels=[['System'],['Zone Name']],
                          labels=[[],[]],
                          names=[u'System', u'Zone Name'])
    zone_info = pd.DataFrame(index=index, columns=zone_info_cols)
    sv_a_dict['Zones'] = zone_info
    
    return sv_a_dict

   
def post_process_sv_a(sv_a_dict, output_to_csv=True):
    """
    Convert the dataframe to numeric dtype 
    and calculates some efficiency metrics, such as Fan W/CFM
    
    Args:
    ------
        sv_a_dict(dict pd.DataFrame): Dictionary of DataFrame with SV-A report data
        
        output_to_csv (boolean): whether you want to output 'SV-A.csv'
        
    Returns:
    --------
        system_info(pd.DataFrame): dataframe in numeric dtype and more metrics
        
        Also spits out a 'SV-A.csv' file if required.
            
        
    Needs:
    -------------------------------
        import pandas as pd
    
    """

    # Convert to numeric
    sv_a_dict['Systems'].ix[:,1:] = sv_a_dict['Systems'].ix[:,1:].apply(lambda x: pd.to_numeric(x))
    
    not_num = ['Fan Placement', 'Fan Control']
    num_cols = [x for x in sv_a_dict['Fans'].columns if x not in not_num]
    sv_a_dict['Fans'][num_cols] = sv_a_dict['Fans'][num_cols].apply(lambda x: pd.to_numeric(x))
    
    sv_a_dict['Zones'] = sv_a_dict['Zones'].apply(lambda x: pd.to_numeric(x))
    
    # Calculate Fan W/CFM
    # At Central level
    sv_a_dict['Fans']['W/CFM'] = sv_a_dict['Fans']['Power Demand (kW)'] * 1000 / sv_a_dict['Fans']['Capacity (CFM)']
    sv_a_dict['Zones']['W/CFM'] = sv_a_dict['Zones']['Fan (kW)'] * 1000 / sv_a_dict['Zones']['Supply Flow (CFM)']
    
    # Output to CSV
    if sys.version_info < (3,0):
        with open('SV-A.csv', 'wb') as f:
            print('SV-A Report\n\n', file=f)
        with open('SV-A.csv', 'ab') as f:
            for k, v in sv_a_dict.items():
                print(k, file=f)
                v.to_csv(f)
                print('',file=f)
    else:
        with open('SV-A.csv', 'w') as f:
            print('SV-A Report\n\n', file=f)
        with open('SV-A.csv', 'a') as f:
            for k, v in sv_a_dict.items():
                print(k, file=f)
                v.to_csv(f)
                print('',file=f)
    
    return sv_a_dict
    
    
def parse_sim(coef_path=None, sim_path=None):
    """
    Loads the SIM file from eQuest and parses out BEPS for unmet hours,
    and PV-A and SV-A (in progress)
    
    Args:
    ------
        
        sim_path (str): Path to the .SIM file generated by eQuest.
            If None, will look one folder up for the 'XXXX - Baseline Design.SIM'
        
        
    Returns:
    --------
        usage(pd.DataFrame): a dataframe with index = 12 months and with MultiIndexed columns:
            * 1st Level is each fuel
            * 2nd Level is either 'Actual' or 'Predicted'
            
        
    Needs:
    -------------------------------
        import pandas as pd
        import re
    
    """
    
    if coef_path is None:
        print("Will only parse out the reports as needed, no weather normalization done, this is an AMY case")
    else:
        # Read the csv
        df = pd.read_csv(coef_path ,index_col=0)
        
        # The numeric (should be the 3 first) are the coefs
        coefs = df.ix[:,df.dtypes == 'float'].T
        
        # Meter name is for correspondance
        meter_correspondance = pd.Series(data=df.index, index=df['MeterName'])
        
        # Create Normalized Actual usage
        usage = pd.DataFrame(np.dot(weather,coefs), index=weather.index, columns=coefs.columns)
        
        # Make it a multiindex
        usage.columns = pd.MultiIndex.from_tuples([(x, 'Actual') for x in usage.columns])
    
    if sim_path is None:
        # list all files with .hsr extension (there should be only one)
        filelist = gb.glob('./*.SIM')
        if len(filelist) != 1:
            print('Sorry but I found several SIM files in the folder. Specify one')
            print(filelist)
            exit
        else:
            sim_path = filelist[0]
            print('Loading {}'.format(sim_path))
    
    # Open the SIM file
    with open(sim_path, encoding="Latin1") as f:
        f_list = f.readlines()
     
    # Initialize test variable for report
    current_report = None

    
    ############################################################################
    #                           PS-B                                           #
    ############################################################################
    meter = None
    
    ############################################################################
    #                        Regexes for BEPU                                  #
    ############################################################################
    unmet_cooling_pattern = re.compile('\s+HOURS ANY ZONE ABOVE COOLING THROTTLING RANGE\s*=\s*([0-9]*)')
    unmet_heating_pattern = re.compile('\s+HOURS ANY ZONE BELOW HEATING THROTTLING RANGE\s*=\s*([0-9]*)')
    
    ############################################################################
    #                   SV-A: System Design Parameters                         #
    ############################################################################
    
    # Initializes a dictionary of dataframes to collect the SV-A report data
    sv_a_dict = create_sv_a_dict()
    sva_header_pattern = 'REPORT- SV-A System Design Parameters for\s+((.*?))\s+WEATHER FILE'
    current_sv_a_section = None
    system_name = None
    
    ############################################################################
    #                                   PV-A                                   #
    ############################################################################
    
    # Initializes a dictionary of dataframes to collect the PV-A report data
    pv_a_dict = create_pv_a_dict()
    current_report = None
    current_plant_equip = None
    plant_equip_pattern = '\*\*\* (.*?) \*\*\*'
    
    
    ############################################################################
    #                            Actual PARSING                                #
    ############################################################################

    for i, line in enumerate(f_list):
        l_list = line.split()
        
        # At least 2 (REPORT- {Name})
        if len(l_list) > 1:
            if l_list[0] == 'REPORT-':
                current_report = l_list[1]
                
                if current_report == 'SV-A':
                    # Match system_name
                    m = re.match(sva_header_pattern, line)
                    if m:
                        system_name = m.group(1)
                    else:
                        print("Error, on line {i} couldn't find the name for the system. Here is the line:".format(i=i))
                        print(line)
                continue     
         
        # Parsing out PS-B (ONLY IF COEF_PATH is supplied)
        if current_report == 'PS-B' and coef_path is not None:
            if len(l_list) > 0:

                if not meter is None:
                    if l_list[0] in ['KWH','THERM']:
                        # Find the proper name of the fuel
                        fuel = meter_correspondance[meter]
                        # Put the monthly value in the right column
                        usage[(meter_correspondance[meter],'Predicted')] = pd.to_numeric(l_list[1:-1])

                # Check if it starts with like 'EM1' or 'FM1' (or something else)
                if l_list[0] in meter_correspondance.index:
                    meter = l_list[0]
                    
        # Parsing out BEPU, right now only unmet heating and cooling hours            
        if current_report == 'BEPU':
            m1 = re.match(unmet_cooling_pattern, line)
            m2 = re.match(unmet_heating_pattern, line)
            if m1:
                unmet_cooling = int(m1.group(1))
                if unmet_cooling > 300:
                    print("/!\ Too many unmet cooling hours: {n}".format(n=unmet_cooling))
                else:
                    print("Unmet Cooling Hours: {n}".format(n=unmet_cooling))
            elif m2:
                unmet_heating = int(m2.group(1))
                if unmet_heating > 300:
                    print("/!\ Too many unmet heating hours: {n}".format(n=unmet_heating))
                else:
                    print("Unmet Heating Hours: {n}".format(n=unmet_heating))
                
        # Parsing out PV-A     
        if current_report == 'PV-A':
            m = re.match(plant_equip_pattern, line)
            if m:
                current_plant_equip = m.group(1)

            # If the line starts with a number or letter a-zA-Z0-9
            if re.match('^\w', line):
                m2 = re.match('^(.*?)\s{2,}', line)
                if m2:
                    equip_name = m2.group(1)
                    pv_a_dict[current_plant_equip].ix[equip_name,:] = re.split('\s{2,}', f_list[i+1].strip())
        
        # Parsing out SV-A
        if current_report == 'SV-A' and len(l_list) > 0:
            
            # Check with section: System, Fan, or Zone
            if l_list[0] in ['SYSTEM','FAN','ZONE']:
                current_sv_a_section = l_list[0]
            
            if current_sv_a_section == 'SYSTEM':
                # If starts by an alpha
                if re.match('^\w', line):
                    sv_a_dict['Systems'].ix[system_name] = l_list
                        
            if current_sv_a_section == 'FAN':
                # If Starts by two spaces and an alpha
                if re.match('^\s{2}\w', line):
                    sv_a_dict['Fans'].ix[(system_name, l_list[0]), :]= l_list[1:]
            
            if current_sv_a_section == 'ZONE':
                if re.match('^\w', line):
                    # Split by at least two spaces (otherwise names of zones like "Apt 1 Zn" becomes three elements in list)
                    l_list = re.split('\s{2,}', line.strip())
                    try:
                        sv_a_dict['Zones'].ix[(system_name, l_list[0]), :]= l_list[1:]             
                    except:
                        print(i)
                        print(line)


    # Post-process SV-A
    sv_a_dict = post_process_sv_a(sv_a_dict, output_to_csv=True)
    
    # Post process PV-A
    # Convert numeric and Calculate some more metrics for the plant equipment (PV-A)
    pv_a_dict = post_process_pv_a(pv_a_dict, output_to_csv=True)
    
    # Sort columns
    if coef_path is None:
        return None, sv_a_dict, pv_a_dict
    else:
        usage = usage.sortlevel(0, axis=1) 
        return usage, sv_a_dict, pv_a_dict
    
    
    
    



################################################################################
#                               RUN FUNCTIONS                                  #
################################################################################ 
    
def custom_apply_zones(x):
    """ Aggregate zone data to the system level
    
    For the zones, some columns should be summed (CFM, Capacity, etc)
    But others should be averaged
    """
    # For these three columns, do a mean
    if x.name in ['Minimum Flow (Frac)', 'Sensible (FRAC)', 'W/CFM']:
        return np.mean(x)
    # For the rest, do a sum
    else:
        return np.sum(x)
    

# If launched rather than imported
if __name__ == "__main__":
    
    # Make it IPython compatible as well.
    try:
        __IPYTHON__
        is_ipython = True
    except NameError:
        is_ipython = False
        # Import the new open function, that supports encoding
        from io import open
    
        # Change cwd to this file, so that the import of the csv in same directory works smoothly
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        os.chdir(dname)
        sys.path.append(os.getcwd())
    
    # Parse the SIM file (Not the PS-B report) to get unmet hours, SV-A and PV-A report data
    _, sv_a_dict, pv_a_dict = parse_sim(coef_path=None, sim_path=None)
    last_print = "In an interactive prompt, the variables 'usage', 'sv_a_dict', 'pv_a_dict' are initialized"
       
    print(last_print)
    
    # Only from command line
    if not is_ipython:
        # Add some plotting to show how it works 
        pv_a_dict['PUMPS'].ix[:, 'W/GPM'].plot(kind='bar', figsize=(16,9), title='Pump W/GPM', color='#EB969C');
        #sns.despine()
        plt.tight_layout()
        plt.show();
        
        sv_a_dict['Fans'].ix[:,'W/CFM'].plot(kind='barh', figsize=(12,10), color='#EB969C', title='Fan W/CFM')
        plt.tight_layout()
        #sns.despine()   
        plt.show();
        
        # Zones: aggregate to system level
        zones = sv_a_dict['Zones']
        print('Here are the first 15 zones, only showing CFM column')
        print(zones['Supply Flow (CFM)'].head(15))
        print("You see above that rows are index by ('System', 'Zone Name'). Now let's aggregate all zones under each system")
        
        # After the groupby, the apply applies to each group dataframe. So I use a lambda x to apply to each column
        zones_agg_metrics = zones.groupby(level='System').apply(lambda x: x.apply(custom_apply_zones))
        # Recalc a weighted W/CFM
        zones_agg_metrics['W/CFM'] = zones_agg_metrics['Fan (kW)'] * 1000 / zones_agg_metrics['Supply Flow (CFM)']
        print("\n\nHere is the result of the aggregation, only showung CFM column again")
        print(zones_agg_metrics['Supply Flow (CFM)'].head())
    
    

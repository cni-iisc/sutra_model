'''
This file collates the regression and plotting modules used in SUTRA implementation.
A brief description of each module is given in the beginning.
All references in this script are with respect to the paper arXiv:2101.09158v4, posted on 27 September 2021.
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, date
from sklearn.linear_model import LinearRegression, Ridge
import sklearn


def prepare_state_frame(start_index_calibration_data, days_calibration):
    """
    Get the dataframe of the reported cases, given a start index and duration.
    Parameters
    ----------
    start_index_calibration_data : int
        Start index of the reported cases (from 02 March 2020).
    days_calibration : int
        Duration (in days) of the reported cases.

    Returns
    -------
    state_frame : dataframe
        The cases time series.

    """
    states = pd.read_csv('target_curves/data.csv')['state'].tolist() 
    smooth_days = 7
    inf = pd.read_csv('target_curves/data.csv')
    
   
    state_frame = pd.DataFrame(columns=['state']+(list(range(0,days_calibration))))
    state_frame['state'] = states
    
    for x in states:
        state_index = inf[inf['state']==x].index.item()
        i_data = (inf.iloc[state_index,1:].values[start_index_calibration_data-smooth_days::].astype(int))
        i_data_average = (np.convolve(i_data, np.ones(smooth_days))/smooth_days)[smooth_days-1::].astype(int).tolist()[0:days_calibration]

        for i in range(days_calibration):
            state_frame.at[state_frame['state']==x,i] = int(i_data_average[i])
    return state_frame


def run_sutra_model(phase_starts, betas, rhos, p0, gamma):
    """
    Implements the SUTRA model equations given in (5)--(7), and the detected trajectory (14).

    Parameters
    ----------
    phase_starts : list
        The list of phase start dates.
    betas : list
        The list of the contact rate parameters for each phase.
    rhos : list
        The list of the reach parameters for each phase.
    p0 : int
        The total population.
    gamma : float
        The recovery rate.

    Returns
    -------
    NT : array
        The time series of the detected trajectory.
    T : array
        The time series of the tested positive trajectory.
    RT : array
        The time series of the recovered from tested trajectory.

    """

    days_plot = (phase_starts[-1] - phase_starts[0]).days + 1
    
    s = np.zeros(days_plot, dtype = 'longdouble')
    u = np.zeros(days_plot, dtype = 'longdouble')
    t = np.zeros(days_plot, dtype = 'longdouble')
    ru = np.zeros(days_plot, dtype = 'longdouble')
    rt = np.zeros(days_plot, dtype = 'longdouble')
    
    NT = np.zeros(days_plot)
    T = np.zeros(days_plot)
    RT = np.zeros(days_plot)
    
    u[0] = 33/p0
    t[0] = 1/p0
    s[0] = 1-(u[0] + t[0] + ru[0] + rt[0])
    
    
    for i in range(len(betas)-1):
        beta = betas[i]
        epsilon = 1/33
        rho = rhos[i]
        if i==0:
            c = 0
        else:
            init_index = phase_indices[-1]
            c = (t[init_index] + u[init_index]) + (rt[init_index] + ru[init_index]) - (1/epsilon)*(t[init_index] + rt[init_index])
        
        phase_length = (phase_starts[i+1] - phase_starts[i]).days
        cumulative_phase_length_previous = (phase_starts[i] - phase_starts[0]).days
        phase_indices = np.array([cumulative_phase_length_previous + k for k in range(1,phase_length+1)])
       
        for j in phase_indices:
            s[j] = s[j-1] - beta*s[j-1]*u[j-1]
            u[j] = u[j-1] + beta*s[j-1]*u[j-1] - epsilon*beta*s[j-1]*u[j-1] - gamma*u[j-1]
            t[j] = t[j-1] + epsilon*beta*s[j-1]*u[j-1] - gamma*t[j-1]
            ru[j] = ru[j-1] + gamma*u[j-1]
            rt[j] = rt[j-1] + gamma*t[j-1]
        
        T[phase_indices] = rho*p0*t[phase_indices]
        RT[phase_indices] = rho*p0*rt[phase_indices]
        
        rho_tilde = rho*epsilon*(1-c)
        beta_tilde = beta*(1-epsilon)*(1-c)
        NT[phase_indices]  = (T[phase_indices] - (1/(rho_tilde*p0))*((T[phase_indices] +RT[phase_indices] )*T[phase_indices] ))*beta_tilde
            
    return NT, T, RT

def run_sutra_model_smooth_transitions(phase_starts, drift_periods, betas, rhos, p0, gamma):
    """
    Implements the SUTRA model (5)--(7) and (14), with smooth parameters across phase boundaries, as explained in Section 7.3.

    Parameters
    ----------  
    phase_starts : list
        The list of phase start dates.
    drift_periods : list
        The list of the duration of the drift periods for each phase.
    betas : list
        The list of the contact rate parameters for each phase.
    rhos : list
        The list of the reach parameters for each phase.
    p0 : int
        The total population.
    gamma : float
        The recovery rate.

    Returns
    -------
    NT : array
        The time series of the detected trajectory.
    T : array
        The time series of the tested positive trajectory.
    RT : array
        The time series of the recovered from tested trajectory.

    """
    
    days_plot = (phase_starts[-1] - phase_starts[0]).days + 1
    
    s = np.zeros(days_plot, dtype = 'longdouble')
    u = np.zeros(days_plot, dtype = 'longdouble')
    t = np.zeros(days_plot, dtype = 'longdouble')
    ru = np.zeros(days_plot, dtype = 'longdouble')
    rt = np.zeros(days_plot, dtype = 'longdouble')
    
    NT = np.zeros(days_plot)
    T = np.zeros(days_plot)
    RT = np.zeros(days_plot)
    
    u[0] = 330/p0
    t[0] = 10/p0
    s[0] = 1-(u[0] + t[0] + ru[0] + rt[0])
    
    
    for i in range(len(betas)-1):
        drift_period_length = drift_periods[i]
                                  
        if i==0:
            c = 0
        else:
            init_index = phase_indices[-1]
            c = (t[init_index] + u[init_index]) + (rt[init_index] + ru[init_index]) - (1/epsilon)*(t[init_index] + rt[init_index])
        
        phase_length = (phase_starts[i+1] - phase_starts[i]).days
        cumulative_phase_length_previous = (phase_starts[i] - phase_starts[0]).days
        phase_indices = np.array([cumulative_phase_length_previous + k for k in range(1,phase_length+1)])
        count = 0
        for j in phase_indices:
            if ((i>0 and drift_period_length > 0) and count<=drift_period_length):
                beta = betas[i-1] * ((betas[i]/betas[i-1])**(count/drift_period_length))
                epsilon = 1/33
                rho = rhos[i-1]* ((rhos[i]/rhos[i-1])**(count/drift_period_length))
            else:
                beta = betas[i]
                epsilon = 1/33
                rho = rhos[i]    

            s[j] = s[j-1] - beta*s[j-1]*u[j-1]
            u[j] = u[j-1] + beta*s[j-1]*u[j-1] - epsilon*beta*s[j-1]*u[j-1] - gamma*u[j-1]
            t[j] = t[j-1] + epsilon*beta*s[j-1]*u[j-1] - gamma*t[j-1]
            ru[j] = ru[j-1] + gamma*u[j-1]
            rt[j] = rt[j-1] + gamma*t[j-1]
            
            count += 1
            T[j] = rho*p0*t[j]
            RT[j] = rho*p0*rt[j]
            
            rho_tilde = rho*epsilon*(1-c)
            beta_tilde = beta*(1-epsilon)*(1-c)
            NT[j]  = (T[j] - (1/(rho_tilde*p0))*((T[j] +RT[j] )*T[j] ))*beta_tilde
            
    return NT, T, RT

def run_sutra_model_smooth_transitions_restarts(phase_starts, drift_periods, restart_indices, betas, rhos, p0, gamma):
    """
    Implements the SUTRA model (5)--(7) and (14), with smooth parameter transitions, and restarts at the beginning of the second and the third waves.  

    Parameters
    ----------
    phase_starts : list
        The list of phase start dates.
    drift_periods : list
        The list of the duration of the drift periods for each phase.
    restart_indices: list 
        The indices of phases at which the model needs to be restarted.
    betas : list
        The list of the contact rate parameters for each phase.
    rhos : list
        The list of the reach parameters for each phase.
    p0 : int
        The total population.
    gamma : float
        The recovery rate.


    Returns
    -------
    NT : array
        The time series of the detected trajectory.
    T : array
        The time series of the tested positive trajectory.
    RT : array
        The time series of the recovered from tested trajectory.

    """
    days_plot = (phase_starts[-1] - phase_starts[0]).days + 1
    
    s = np.zeros(days_plot, dtype = 'longdouble')
    u = np.zeros(days_plot, dtype = 'longdouble')
    t = np.zeros(days_plot, dtype = 'longdouble')
    ru = np.zeros(days_plot, dtype = 'longdouble')
    rt = np.zeros(days_plot, dtype = 'longdouble')
    
    NT = np.zeros(days_plot)
    T = np.zeros(days_plot)
    RT = np.zeros(days_plot)
    
    u[0] = 100/p0
    t[0] = 10/p0
    s[0] = 1-(u[0] + t[0] + ru[0] + rt[0])
    
    
    for i in range(len(betas)-1):
        if i == restart_indices[0]:
            ru[j] = 0.22
            rt[j] = 0.06
            s[j] = 1-(u[j] + t[j] + ru[j] + rt[j])
        elif i==restart_indices[1]:
            ru[j] = 0.65
            rt[j] = 0.1
            t[j] = 75/p0
            u[j] = 33*75/p0
            s[j] = 1-(u[j] + t[j] + ru[j] + rt[j])
        drift_period_length = drift_periods[i]
                                  
        if i==0:
            c = 0
        else:
            init_index = phase_indices[-1]
            c = (t[init_index] + u[init_index]) + (rt[init_index] + ru[init_index]) - (1/epsilon)*(t[init_index] + rt[init_index])
        
        phase_length = (phase_starts[i+1] - phase_starts[i]).days
        cumulative_phase_length_previous = (phase_starts[i] - phase_starts[0]).days
        phase_indices = np.array([cumulative_phase_length_previous + k for k in range(1,phase_length+1)])
        count = 0
        for j in phase_indices:
            if ((i>0 and drift_period_length > 0) and count<=drift_period_length):
                beta = betas[i-1] * ((betas[i]/betas[i-1])**(count/drift_period_length))
                epsilon = 1/33
                rho = rhos[i-1]* ((rhos[i]/rhos[i-1])**(count/drift_period_length))
            else:
                beta = betas[i]
                epsilon = 1/33
                rho = rhos[i]    

            s[j] = s[j-1] - beta*s[j-1]*u[j-1]
            u[j] = u[j-1] + beta*s[j-1]*u[j-1] - epsilon*beta*s[j-1]*u[j-1] - gamma*u[j-1]
            t[j] = t[j-1] + epsilon*beta*s[j-1]*u[j-1] - gamma*t[j-1]
            ru[j] = ru[j-1] + gamma*u[j-1]
            rt[j] = rt[j-1] + gamma*t[j-1]
            
            count += 1
            T[j] = rho*p0*t[j]
            RT[j] = rho*p0*rt[j]
            
            rho_tilde = rho*epsilon*(1-c)
            beta_tilde = beta*(1-epsilon)*(1-c)
            NT[j]  = (T[j] - (1/(rho_tilde*p0))*((T[j] +RT[j] )*T[j] ))*beta_tilde
            
    return NT, T, RT

def create_plots(phase_starts, start_date, NT, days_data):
    """
    Plotting script.

    Parameters
    ----------
    phase_starts : TYPE
        DESCRIPTION.
    start_date : date
        The start date of the model and data.
    NT : array
        The array of the detected trajectory time series from the model.
    days_data : int
        The number of days for which the reported cases data needs to be plotted.

    Returns
    -------
    None.

    """
    x_dates = phase_starts
    x_indices = np.array([(j - start_date).days for j in x_dates])
    x_labels = [j.strftime('%d/%m/%Y') for j in x_dates]
    
    
    plt.rcParams['figure.figsize'] = [10, 6]
    state_frame_plot = prepare_state_frame(7,days_data)
    
    x = 'India'
    curve1= np.convolve(NT, np.ones(7))/7
    curve_data = state_frame_plot.loc[state_frame_plot['state']==x].values[0][1+x_indices[0]::]
    
    x_indices = x_indices - x_indices[0]

    plt.plot((curve1), 'g^--',label='Simulation')
    plt.plot((curve_data), 'ro-',label='Target')
    
    plt.xticks(x_indices, x_labels, rotation='vertical')
    plt.grid(True)
    plt.ylabel('Daily cases')
    plt.title(x+ ' -- Daily') 
    plt.legend()
    #plt.savefig('./plots/daily_'+x, bbox_inches='tight')    
    plt.show()
    plt.close()

def run_sutra_model_restart(phase_starts, betas, rhos, p0, gamma,restart_phase_index):
    """
    Implements the SUTRA model (5)--(7) and (14), and restarts at the beginning of the second wave.
    
    Parameters
    ----------
    phase_starts : list
        The list of phase start dates.
    betas : list
        The list of the contact rate parameters for each phase.
    rhos : list
        The list of the reach parameters for each phase.
    p0 : int
        The total population.
    gamma : float
        The recovery rate.
    restart_phase_index: int
        The index of the phase at which the model needs to be restarted.


    Returns
    -------
    NT : array
        The time series of the detected trajectory.
    T : array
        The time series of the tested positive trajectory.
    RT : array
        The time series of the recovered from tested trajectory.

    """
    days_plot = (phase_starts[-1] - phase_starts[0]).days + 1
    
    s = np.zeros(days_plot, dtype = 'longdouble')
    u = np.zeros(days_plot, dtype = 'longdouble')
    t = np.zeros(days_plot, dtype = 'longdouble')
    ru = np.zeros(days_plot, dtype = 'longdouble')
    rt = np.zeros(days_plot, dtype = 'longdouble')
    
    NT = np.zeros(days_plot)
    T = np.zeros(days_plot)
    RT = np.zeros(days_plot)
    
    u[0] = 33/p0
    t[0] = 1/p0
    s[0] = 1-(u[0] + t[0] + ru[0] + rt[0])
    
    
    for i in range(len(betas)-1):
        if i==restart_phase_index:
            ru[j] = 0.2
            rt[j] = 0.1
            s[j] = 1-(u[j] + t[j] + ru[j] + rt[j])
            
        beta = betas[i]
        epsilon = 1/33
        rho = rhos[i]
        if i==0:
            c = 0
        else:
            init_index = phase_indices[-1]
            c = (t[init_index] + u[init_index]) + (rt[init_index] + ru[init_index]) - (1/epsilon)*(t[init_index] + rt[init_index])
        
        phase_length = (phase_starts[i+1] - phase_starts[i]).days
        cumulative_phase_length_previous = (phase_starts[i] - phase_starts[0]).days
        phase_indices = np.array([cumulative_phase_length_previous + k for k in range(1,phase_length+1)])
        for j in phase_indices:
            s[j] = s[j-1] - beta*s[j-1]*u[j-1]
            u[j] = u[j-1] + beta*s[j-1]*u[j-1] - epsilon*beta*s[j-1]*u[j-1] - gamma*u[j-1]
            t[j] = t[j-1] + epsilon*beta*s[j-1]*u[j-1] - gamma*t[j-1]
            ru[j] = ru[j-1] + gamma*u[j-1]
            rt[j] = rt[j-1] + gamma*t[j-1]
        
        T[phase_indices] = rho*p0*t[phase_indices]
        RT[phase_indices] = rho*p0*rt[phase_indices]
        
        rho_tilde = rho*epsilon*(1-c)
        beta_tilde = beta*(1-epsilon)*(1-c)
    
        NT[phase_indices]  = (T[phase_indices] - (1/(rho_tilde*p0))*((T[phase_indices] +RT[phase_indices] )*T[phase_indices] ))*beta_tilde
    
    return NT, T, RT

def create_beta_rho(betatilde, rhotilde, phase_starts, phase_lengths):
    """
    Create betatilde and rhotilde array, given the parameters for each phase.

    Parameters
    ----------
    betatilde : list
        The list of the betatilde parameter for each phase.
    rhotilde : list
        The list of the rhotilde parameter for each phase.
    phase_starts : list
        The list of phase start dates.
    phase_lengths : list
        The list of phase lengths.

    Returns
    -------
    betatilde_t : array
        The per-day array of the betatilde parameter.
    rhotilde_t : array
        The per-day array of the rhotilde parameter.

    """
    betatilde_t = np.zeros(np.sum(phase_lengths))
    rhotilde_t = np.zeros(np.sum(phase_lengths))

    for i in range(len(phase_starts)):
        start_offset = (phase_starts[i] - phase_starts[0]).days
        for j in range(phase_lengths[i]):
            betatilde_t[j + start_offset] = betatilde[i]
            rhotilde_t[j + start_offset] = rhotilde[i]
    return betatilde_t, rhotilde_t

def create_beta_rho_smooth(betatilde, rhotilde, phase_starts, drift_periods, phase_lengths):
    """
    Create betatilde and rhotilde array, smoothended according to Section 7.3, given the parameters for each phase.  

    Parameters
    ----------
    betatilde : list
        The list of the betatilde parameter for each phase.
    rhotilde : list
        The list of the rhotilde parameter for each phase.
    phase_starts : list
        The list of phase start dates.
    phase_lengths : list
        The list of phase lengths.
    drift_periods : list
        The list of the duration of the drift periods for each phase.

    Returns
    -------
    betatilde_t : array
        The per-day array of the betatilde parameter after smoothening.
    rhotilde_t : array
        The per-day array of the rhotilde parameter after smoothening.
    """
    
    betatilde_t = np.zeros(np.sum(phase_lengths))
    rhotilde_t = np.zeros(np.sum(phase_lengths))

    for i in range(len(phase_starts)):
        start_offset = (phase_starts[i] - phase_starts[0]).days
        drift_period_length = drift_periods[i]
        for j in range(phase_lengths[i]):
            if j<=drift_period_length and drift_period_length > 0 and i >0:
                betatilde_t[j + start_offset] = betatilde[i-1]*((betatilde[i]/betatilde[i-1])**(j/drift_period_length))
                rhotilde_t[j + start_offset] = rhotilde[i-1]*((rhotilde[i]/rhotilde[i-1])**(j/drift_period_length))
            else:
                betatilde_t[j + start_offset] = betatilde[i]
                rhotilde_t[j + start_offset] = rhotilde[i]
    return betatilde_t, rhotilde_t

# 
def create_detected_trajectory(betatilde_t, rhotilde_t, T0, RT0, p0, gamma):
    """
    Create the detected trajectory; implements the recursion in Lemma 5.

    Parameters
    ----------
    betatilde_t : array
        The per-day array of the betatilde parameter.
    rhotilde_t : array
        The per-day array of the rhotilde parameter.
    T0 : int
        The initial condition for the tested varialbe (T).
    RT0 : int
        The initial condition for the recovered from tested varialbe (RT).
    p0 : int
        The total population.
    gamma : float
        The recovery rate.

    Returns
    -------
    NT: array
        The time series of the detected trajectory.
    T : array
        The time series of the tested positive trajectory.
    RT : array
        The time series of the recovered from tested trajectory.

    """
    duration = len(betatilde_t)
    NT = np.zeros(duration+1)
    RT = np.zeros(duration+1)
    T = np.zeros(duration+1)
    
    T[0] = T0
    RT[0] = RT0
    
    for i in range(len(betatilde_t)):
        NT[i+1] = betatilde_t[i]*T[i] - (betatilde_t[i]/(p0*rhotilde_t[i]))*((T[i]+RT[i])*T[i])
        RT[i+1] = RT[i] + gamma*T[i]
        T[i+1] = T[i] + RT[i] + NT[i+1] - RT[i+1]
    return np.maximum(NT, 0), T, RT

def get_data(start_date, duration):
    """
    Get data for plots.

    Parameters
    ----------
    start_date : date
        Start date of the data.
    duration : int
        Duratin of the data.

    Returns
    -------
        array
        The array of the reported cases.

    """
    df = pd.read_csv('target_curves/data.csv')
    data_start_date = date(2020,3,2)
    start_index = (start_date-data_start_date).days
    return df.loc[df['state']=='India'].values[0][1::][start_index:start_index+duration].astype(int)

# 
def compute_t_rt(NT, gamma, duration, T0, RT0):
    """
    Compute the T and RT trajectories for regression, as required in the beginnig of Section 7.2.  

    Parameters
    ----------
    NT : array
        The time series of the reported cases data.
    gamma : float
        The recovery rate.
    duration : int
        The duration of which T and RT are to be computed.
    T0 : int
        The initial condition for the tested positive variable.
    RT0 : int
        The initial condition for the recovered from tested positive variable.

    Returns
    -------
    T : array
        The time series of the tested positive trajectory.
    RT : array
        The time series of the recovered from tested trajectory.

    """
    T = np.zeros(duration+1)
    RT = np.zeros(duration+1)
    T[0] = T0
    RT[0] = RT0
    for i in range(1,duration+1):
        T[i] = T[i-1] + NT[i-1] -gamma*T[i-1]
        RT[i] = RT[i-1] + gamma*T[i-1]
    T = T[1::]
    RT = RT[1::]
    return T, RT


def regress_betatilde_rhotilde(NT, T, RT, p0):
    """
    Standard linear regression for rhotilde and betatilde; implements the linear regression of Section 7.2.1.

    Parameters
    ----------
    NT : array
        The time series of the reported cases data.
    T : array
        The time series of the tested positive trajectory.
    RT : array
        The time series of the recovered from tested trajectory.
    p0 : int
        The total population.

    Returns
    -------
    betatilde: float
        The betatilde parameter from regression.
    rhotilde: float
        The rhotilde parameter from regression.
    r2: float
        The R^2 value from regression.
    """
    u = np.convolve(T, np.ones(7), mode='valid')
    v = np.convolve(NT[1::], np.ones(7), mode='valid')
    w = np.convolve((T+RT)*T, np.ones(7), mode='valid')/p0

    X = np.zeros((2, len(u)))
    X[0] = v
    X[1] = w
    X = np.transpose(X)

    reg = LinearRegression(fit_intercept=False, positive=True).fit(X, u)

    betatilde = np.minimum(1/reg.coef_[0],1)
    rhotilde = 1/reg.coef_[1]
    
    r2 = sklearn.metrics.r2_score(u,np.dot(X, [reg.coef_[0], reg.coef_[1]]))
    
    return betatilde, rhotilde, r2

def regress_betatilde_rhotilde_ridge(NT, T, RT, p0):
    """
    Ridge regression to fit the betatilde and rhotilde parameters.

    Parameters
    ----------
    NT : array
        The time series of the reported cases data.
    T : array
        The time series of the tested positive trajectory.
    RT : array
        The time series of the recovered from tested trajectory.
    p0 : int
        The total population.

    Returns
    -------
    betatilde: float
        The betatilde parameter from regression.
    rhotilde: float
        The rhotilde parameter from regression.
    r2: float
        The R^2 value from regression.
    """
    u = np.convolve(T, np.ones(7), mode='valid')
    v = np.convolve(NT[1::], np.ones(7), mode='valid')
    w = np.convolve((T+RT)*T, np.ones(7), mode='valid')/p0

    X = np.zeros((2, len(u)))
    X[0] = v
    X[1] = w
    X = np.transpose(X)

    reg = Ridge(alpha = 1).fit(X, u)
    
    betatilde = 1/reg.coef_[0]
    rhotilde = 1/reg.coef_[1]
    
    r2 = sklearn.metrics.r2_score(u,np.dot(X, [reg.coef_[0], reg.coef_[1]]) + reg.intercept_)
    
    return betatilde, rhotilde, r2

def regress_betatilde_rhotilde_using_minimize(NT, T, RT, p0):  
    """
    Another regression script that uses scipy.minimize.

    Parameters
    ----------
    NT : array
        The time series of the reported cases data.
    T : array
        The time series of the tested positive trajectory.
    RT : array
        The time series of the recovered from tested trajectory.
    p0 : int
        The total population.

    Returns
    -------
    betatilde: float
        The betatilde parameter from regression.
    rhotilde: float
        The rhotilde parameter from regression.
    r2: float
        The R^2 value from regression.
    """
    def errorfun(params):
        u = np.convolve(T, np.ones(7), mode='valid')
        v = np.convolve(NT[1::], np.ones(7), mode='valid')
        w = np.convolve((T+RT)*T, np.ones(7), mode='valid')/p0
        
        return np.sqrt(np.sum(np.square(np.abs(u - v/params[0] - w/params[1]))))
    
    res = minimize(errorfun, (0.01,0.01), bounds = [(0,1), (0,1/33)])
    
    r2 = 1 - np.square(errorfun(res.x))/np.sum(np.square(np.convolve(T, np.ones(7), mode='valid')))
    return res.x[0], res.x[1], r2

def phase_detection_algorithm(gamma, end_date):
    """
    Phase detection algorithm explained in Section 7.2.3 -- work in progress.

    Parameters
    ----------
    gamma : float
        The recovery rate.
    end_date : date
        The end date for the phase detection algorithm.

    Returns
    -------
    None.

    """
    start_date = date(2020,3,2)
    betatilde_array = []
    rhotilde_array = []
    r2_array = []
    r2 = 1
    T0 = 5
    RT0 = 0
    remaining_days = True
    
    while remaining_days ==True:
        phase_length = 0
        duration = 5
        r2 = 1
        while r2>0.98:
            NT = get_data(start_date, duration + 7+1)
            T, RT = compute_t_rt(NT, gamma, duration+7, T0, RT0)
            betatilde, rhotilde, r2 = regress_betatilde_rhotilde(NT, T, RT, p0)
            T0 = T[-1]
            RT0 = RT[-1]
            phase_length += 1
            duration += 1
            
        betatilde_array.append(betatilde)
        rhotilde_array.append(rhotilde)
        r2_array.append(r2)
        start_date = start_date + datetime.timedelta(phase_length)
        if (end_date - start_date).days <1:
            remaining_days = False

def regress(gamma,p0,cir,phase_starts, T, RT):
    """
    The script that runs the regression for all phases.

    Parameters
    ----------
    gamma : float
        The recovery rate.
    p0 : int
        The total population.
    cir : float
        The epsilon parameter.
    phase_starts : list
        The list of phase start dates.
    T : array
        The time series of the tested positive trajectory.
    RT : array
        The time series of the recovered from tested trajectory.

    Returns
    -------
    df : dataframe
        Regression outcomes.

    """
    betatilde = np.zeros(len(phase_starts))
    rhotilde = np.zeros(len(phase_starts))
    r2 = np.zeros(len(phase_starts))
    for i in range(len(phase_starts)-1):
        start_date = phase_starts[i]
        duration = (phase_starts[i+1] - phase_starts[i]).days
        NT = get_data(start_date, duration + 7+1)

        T0 = T[-1]
        RT0 = RT[-1]
        T, RT = compute_t_rt(NT, gamma, duration+7, T0, RT0)
        betatilde[i], rhotilde[i], r2[i] = regress_betatilde_rhotilde_ridge(NT, T, RT, p0)

    df = pd.DataFrame(columns = ['Phase Start', 'betatilde', 'rhotilde (%)', 'beta (approx)', 'rho (approx)'])

    df['Phase Start'] = phase_starts[:-1]
    df['betatilde'] = betatilde[:-1]
    df['rhotilde (%)'] = rhotilde[:-1]*100
    df['beta (approx)']  = betatilde[:-1]/(1-cir)
    df['rho (approx)'] = (rhotilde[:-1]*100)/cir
    df['r2'] = r2[:-1]
    return df

def convert_for_forecast_hub(output, start_date, todays_date):
    """
    Convert the detected trajectory for forecast hub.

    Parameters
    ----------
    output : array
        The array of predictions.
    start_date : date
        The start date for the data.
    todays_date : date
        The date on which the predictions are computed.

    Returns
    -------
    None.

    """

    days_offset = (todays_date - start_date).days
    predict_forward = 4

    file_out = pd.DataFrame(columns = ['avl_data', 'fct_date', 'fct_lb', 'fct_std',  \
                            'fct_ub', 'horizon', 'location', 'method', 'sig', \
                             'step_ahead', 'TRUE', 'value', 'wts'])

    x = 'India'
    for y in range(predict_forward):
        start_column =  y*7 + days_offset
        end_column = (y+1)*7 + days_offset

        val = int(np.sum([output[i] for i in range(start_column, end_column)]))
        val_lb = val# int(np.sum([output_lb.loc[output_lb[0]==x,i].item() for i in range(start_column, end_column+1)])/cir)
        val_ub = val #int(np.sum([output_ub.loc[output_ub[0]==x,i].item() for i in range(start_column, end_column+1)])/cir)
        new_row = { 'avl_data': todays_date, \
                    'fct_date': todays_date+datetime.timedelta(days = 7*(y+1)), \
                    'fct_lb': val_lb, \
                    'fct_std': 0 ,  \
                    'fct_ub': val_ub, \
                    'horizon' : y, \
                    'location': x, \
                    'method':'Omicron', \
                    'sig': ' ', \
                    'step_ahead':  str(y+1)+'-step_ahead', \
                    'TRUE': ' ', \
                    'value': val, \
                    'wts': ' '}
        file_out = file_out.append(new_row, ignore_index = True)

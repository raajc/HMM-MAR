%% TMS-EEG HMM Analysis
%% Loading Brainstorm Data
bst_source_data = {musc_m002_lmc musc_m003_lmc musc_m004_lmc musc_m005_lmc ...
    musc_m008_lmc musc_m009_lmc musc_m010_lmc musc_m011_lmc musc_m012_lmc ...
    musc_m013_lmc musc_m015_lmc musc_m016_lmc musc_m017_lmc musc_m018_lmc ...
    musc_m019_lmc musc_m020_lmc musc_m021_lmc yd_c001_lmc yd_c004_lmc ...
    yd_c007_lmc yd_c017_lmc yd_c023_lmc yd_c025_lmc yd_c029_lmc yd_c030_lmc ...
    yd_c037_lmc yd_c038_lmc yd_c041_lmc yd_c043_lmc yd_c045_lmc yd_c049_lmc ...
    yd_c050_lmc yd_c053_lmc yd_c055_lmc yd_c056_lmc};
%% Rearranging DK Data so it matches eventual mapping template 
for subji = 1:size(bst_source_data,2)
    
    orig_data = bst_source_data(subji);
    orig_data = orig_data{1,1}.Value'; % transposing so its in time by source 
    temp_data= zeros(size(orig_data)); 

    temp_data(:,1) = orig_data(:,1);
    temp_data(:,2) = orig_data(:,3);
    temp_data(:,3) = orig_data(:,5);
    temp_data(:,4) = orig_data(:,7);
    temp_data(:,5) = orig_data(:,9);
    temp_data(:,6) = orig_data(:,13);
    temp_data(:,7) = orig_data(:,15);
    temp_data(:,8) = orig_data(:,17);
    temp_data(:,9) = orig_data(:,21);
    temp_data(:,10) = orig_data(:,23);
    temp_data(:,11) = orig_data(:,25);
    temp_data(:,12) = orig_data(:,27);
    temp_data(:,13) = orig_data(:,29);
    temp_data(:,14) = orig_data(:,31);
    temp_data(:,15) = orig_data(:,35);
    temp_data(:,16) = orig_data(:,33);
    temp_data(:,17) = orig_data(:,37);
    temp_data(:,18) = orig_data(:,39);
    temp_data(:,19) = orig_data(:,41);
    temp_data(:,20) = orig_data(:,43);
    temp_data(:,21) = orig_data(:,45);
    temp_data(:,22) = orig_data(:,47);
    temp_data(:,23) = orig_data(:,49);
    temp_data(:,24) = orig_data(:,51);
    temp_data(:,25) = orig_data(:,53);
    temp_data(:,26) = orig_data(:,55);
    temp_data(:,27) = orig_data(:,57);
    temp_data(:,28) = orig_data(:,59);
    temp_data(:,29) = orig_data(:,61);
    temp_data(:,30) = orig_data(:,63);
    temp_data(:,31) = orig_data(:,11);
    temp_data(:,32) = orig_data(:,65);
    temp_data(:,33) = orig_data(:,67);
    temp_data(:,34) = orig_data(:,19);
    temp_data(:,35) = orig_data(:,2);
    temp_data(:,36) = orig_data(:,4);
    temp_data(:,37) = orig_data(:,6);
    temp_data(:,38) = orig_data(:,8);
    temp_data(:,39) = orig_data(:,10);
    temp_data(:,40) = orig_data(:,14);
    temp_data(:,41) = orig_data(:,16);
    temp_data(:,42) = orig_data(:,18);
    temp_data(:,43) = orig_data(:,22);
    temp_data(:,44) = orig_data(:,24);
    temp_data(:,45) = orig_data(:,26);
    temp_data(:,46) = orig_data(:,28);
    temp_data(:,47) = orig_data(:,30);
    temp_data(:,48) = orig_data(:,32);
    temp_data(:,49) = orig_data(:,36);
    temp_data(:,50) = orig_data(:,34);
    temp_data(:,51) = orig_data(:,38);
    temp_data(:,52) = orig_data(:,40);
    temp_data(:,53) = orig_data(:,42);
    temp_data(:,54) = orig_data(:,44);
    temp_data(:,55) = orig_data(:,46);
    temp_data(:,56) = orig_data(:,48);
    temp_data(:,57) = orig_data(:,50);
    temp_data(:,58) = orig_data(:,52);
    temp_data(:,59) = orig_data(:,54);
    temp_data(:,60) = orig_data(:,56);
    temp_data(:,61) = orig_data(:,58);
    temp_data(:,62) = orig_data(:,60);
    temp_data(:,63) = orig_data(:,62);
    temp_data(:,64) = orig_data(:,64);
    temp_data(:,65) = orig_data(:,12);
    temp_data(:,66) = orig_data(:,66);
    temp_data(:,67) = orig_data(:,68);
    temp_data(:,68) = orig_data(:,20);
    
    source_data{subji,1} = temp_data; 
end
    
%% Rearranging Schaefer 100 Parcel Data so it matches eventual mapping template 

% % Set up the Import Options and import the data
% opts = spreadsheetImportOptions("NumVariables", 1);
% 
% % Specify sheet and range
% opts.Sheet = "Schaefer2018_100Parcels_17Netwo";
% opts.DataRange = "A1:A100";
% 
% % Specify column names and types
% opts.VariableNames = "ParcelName";
% opts.VariableTypes = "string";
% 
% % Specify variable properties
% opts = setvaropts(opts, "ParcelName", "WhitespaceRule", "preserve");
% opts = setvaropts(opts, "ParcelName", "EmptyFieldRule", "auto");
% 
% % Import the data
% Schaefer2018100Parcels17Networksorderedit = readtable("/Users/prabhjotdhami/Desktop/HMM/Source/Schaefer2018_100Parcels_17Networks_order_edit.xlsx", opts, "UseExcel", false);
% % Clear temporary variables
% clear opts
% 
% for subji = 1:size(bst_source_data,2)
%     
%     cursubjbst = bst_source_data{subji}; 
%     
%     orig_data = bst_source_data(subji);
%     orig_data = orig_data{1,1}.Value'; % transposing so its in time by source 
%     temp_data= zeros(size(orig_data)); 
% 
%     for parceli = 1:100
%         curparcelname = Schaefer2018100Parcels17Networksorderedit.ParcelName(parceli); 
%         if startsWith(curparcelname, 'LH_')
%             curparcelname = char(strrep(curparcelname, 'LH_', '')); 
%             curparcelname = [curparcelname ' L']; 
%         elseif startsWith(curparcelname, 'RH_')
%             curparcelname = char(strrep(curparcelname, 'RH_', '')); 
%             curparcelname = [curparcelname ' R']; 
%         end
%         
%         temp_data(:,parceli) = orig_data(:, contains({cursubjbst.Atlas.Scouts.Label}, curparcelname)); 
%     end
%         
%     source_data{subji,1} = temp_data; 
% end
%% Load TMS-EEG Data and configure T at original data sample (i.e. 1000 Hz)
%source_data = myHMM_data; 

timepoints = 2000; 

T = cell(size(source_data,1),1);

for subji = 1:size(source_data,1) 
    subji_trial_count = size(source_data{subji,1},1);
    subji_trial_count = subji_trial_count / timepoints;
    cursubjtrial = repmat(timepoints,subji_trial_count,1); 
    T{subji,1} = cursubjtrial;
end
%% Changing format of data for downsampling to work
%source_data = cat(1, source_data{:});
%T = cat(1,T{:});
%% APPLYING FLIP FUNCTION
options = [];
%options.noruns = 1;
%options.maxlag = 29;
options.verbose = 1;
[flips,scorepath] = findflip(source_data,T, options);
source_data = flipdata(source_data,T,flips);
%% Performing HMM

K = 8;

hmm_data = source_data;

options = [];
options.K = K;
options.order = 0;
options.embeddedlags = -29:29;
options.covtype = 'full';
options.zeromean= 1; % 0 is mean, 1 is functional connectivity 
options.standardise = 1;
%options.filter = [4 45];
options.pca = size(hmm_data{1},2) * 2; 
%options.downsample = 100;
options.Fs = 1000; 
%options.useParallel = 0; 
options.verbose = true; 
options.onpower = 0;
options.leakagecorr= -1; 
%newfs = options.downsample; 

% Importantly, if stochastic inference is to be used, the inputs (both data and T; see above)
% must be specified as a cell, with one element per subject. That is, data{1} will be a matrix 
% with the data for subject 1, and T{1} will contain a vector with the trial lengths for subject 1 
% (or just length(data{1}) if this is continuous data with no breaks).

options.BIGNbatch = 5;
%options.BIGNinitbatch = 5;

[hmm, Gamma, Xi, vpath, GammaInit, residuals, fehist] = hmmmar(hmm_data,T,options); 
%% If data is in array format for resampling

% %Resampling T
% [~,T_new] = downsampledata(source_data,T,newfs,1000);
% T = T_new; 
% total_trial = size(T,1); 
% 
% % Creating stimulus array
% stimulus = zeros(newfs*2,1);
% stimulus(newfs,1) = true; 
% stimulus = repmat(stimulus, total_trial, 1);
% 
% myT = T; 
%% If data is in cell format/not resampled 
total_trial = 0;
for i = 1:size(T,1)
    cur_trial_num = size(T{i,1},1); 
    total_trial = total_trial + cur_trial_num;
end

stimulus = zeros(2000,1);
stimulus(1000,1) = true; 
stimulus = repmat(stimulus, total_trial, 1);

myT = cat(1, T{:});
%% Buffering GAMMA 
Gamma = padGamma(Gamma,T,hmm.train);
%% Running EvokedProbability 
timearoundstim = 1.6;

[evokedGamma, t] = evokedStateProbability(stimulus, myT, Gamma, timearoundstim,options);
%% Plotting Individual Evoked States
numofstates = K;

numtoplot = ceil(sqrt(numofstates));

figure;

for statei = 1:numofstates
    subplot(numtoplot,numtoplot,statei);
    plot(evokedGamma(:,statei),'LineWidth',2); 
    mygca = gca();
    mygca.XTick = [0 400 800 1200 1600]; 
    mygca.XTickLabel = {'-800', '-400', '0', '400', '800'};
    grid on; 
    title(['State:' num2str(statei)]); 
    ylabel('Probability')
    xlabel('Time (ms)');
    xlim([0 1600]); 
    vline(timearoundstim*1000/2, '--k');
end

%% Plotting Epochs
figure
t = 500:1500;
subplot(3,1,1)
area(t/1000,Gamma(t,:),'LineWidth',2);  xlim([t(1)/1000 t(end)/1000])
xlabel('Time'); ylabel('State probability')
title('TDE-HMM' )

t = 10501:11500;
subplot(3,1,2)
area(t/1000,Gamma(t,:),'LineWidth',2);  xlim([t(1)/1000 t(end)/1000])
xlabel('Time'); ylabel('State probability')
title('TDE-HMM' )

t = 20501:21500;
subplot(3,1,3)
area(t/1000,Gamma(t,:),'LineWidth',2);  xlim([t(1)/1000 t(end)/1000])
xlabel('Time'); ylabel('State probability')
title('TDE-HMM' )
%% ERP (Evoked Response Probability) of Individual Subjects

% subject_erp = cell(size(hmm_data,1),1);
% 
% curtime1 = 0;
% for subji = 1:size(hmm_data,1)
% 
%     curtime2 = curtime1 + size(hmm_data{subji,:},1);
%     cursubjdata = Gamma((curtime1+1):curtime2,1:K); 
%     curT = size(T{subji},1);
%     
%     tempdata = zeros(2000, K, curT); 
%     temptime1 = 0;
%     for ti = 1:curT
%         
%         temptime2 = (temptime1) + 2000;
%         tempdata(1:2000, 1:K, ti) = cursubjdata(temptime1+1:temptime2,:); 
%         
%         temptime1 = temptime2;
%     end
%    
%     subject_erp(subji,1) = {mean(tempdata,3)}; 
%     
%     curtime1 = curtime2; 
% end
% 
% % Plotting individual data for each state
% for ki = 1:K
%     figure;
%     for subji = 1:size(hmm_data,1)
%         plot(subject_erp{subji,1}(:,ki));
%         hold on
%     end
%     title(['State:' num2str(ki)]);
% end
%     
% % Double Checking State ERPs
% state_erp_GA = cell(K,1);
% for ki = 1:K
%     curstatedata = zeros(2000, size(hmm_data,1)); 
%     for subji = 1:size(hmm_data,1)
%         curstatedata(:,subji) = subject_erp{subji,1}(:,ki);
%     end
%     
%     state_erp_GA{ki,1} = mean(curstatedata,2); 
% end
% %% Baseline vs Post TMS Statistics Using FT
% 
% statelabels = {'State1', 'State2', 'State3', 'State4', 'State5', 'State6','State7', 'State8'};
% 
% % load dummy FT
% subject_erp_FT = cell(size(hmm_data,1),1); 
% subject_bl_FT = cell(size(hmm_data,1),1); 
% 
% for subji = 1:size(subject_erp_FT,1)
%     curstruct = dataGoAngryTimeLock;
%     
%     %cleaning structure
%     curstruct.avg = [];
%     curstruct.var = [];
%     curstruct.time = [];
%     
%     % Adding/editing fields
%     curstruct.dimord = 'chan_time';
%     curstruct.elec.elecpos(K+1:end,:) = []; 
%     curstruct.elec.label(K+1:end) = [];
%     curstruct.elec.pnt(K+1:end,:) = [];
%     curstruct.label = statelabels;
%     curstruct.label = reshape(curstruct.label, 1, K); 
%     curstruct.elec.label = curstruct.label; 
%     curstruct.time = [-1:0.001:0.999]; 
%     curstruct.filename = ['Subject:' num2str(subji)];
%     
%     % Checking data
%     curstruct.avg = subject_erp{subji,1}';
%     [timelock] = ft_datatype_timelock(curstruct); 
%     [timelock] = ft_checkdata(timelock);
%     
%     % Adding baseline
%     cfg = [];
%     cfg.latency = [-0.6 -0.1]; %edit
%     [bl_data] = ft_selectdata(cfg, timelock);
%     bl_data.time = [0:0.001:0.5]; %edit
%     
%     subject_bl_FT{subji,1} = bl_data;
%     
%     % Adding post TMS
%     cfg = [];
%     cfg.latency = [0 0.5]; %edit
%     [erp_data] = ft_selectdata(cfg, timelock);
%     
%     subject_erp_FT{subji,1} = erp_data;
% end
% %% FT Stats
% 
% % Make sure to remove OSL here and add FT instead
% 
% % With cfg.statistic = ‘ft_statfun_actvsblT’, we choose the so-called 
% % activation-versus-baseline T-statistic. This statistic compares the power 
% % in every sample (i.e., a (channel,frequency,time)-triplet) in the activation 
% % period with the corresponding time-averaged power 
% % (i.e., the average over the temporal dimension) in the baseline period. 
% % The comparison of the activation and the time-averaged baseline power is 
% % performed by means of a dependent samples T-statistic.
% 
% cfg = [];
% %cfg.latency = [0 0.5];
% cfg.parameter = 'avg';
% cfg.statistic = 'ft_statfun_actvsblT';
% cfg.method = 'montecarlo';
% cfg.correctm = 'cluster';
% cfg.numrandomization = 10000;
% cfg.computeprob = 'yes';
% cfg.tail = 0;
% cfg.alpha = 0.05;
% cfg.ivar = 1;
% cfg.uvar = 2;
% cfg.design(1,:) = [ones(1,30) ones(1,30)*2];
% cfg.design(2,:) = [1:30 1:30];
% cfg.channel = {'State8'};
%     
% [stat] = ft_timelockstatistics(cfg, subject_bl_FT{:}, subject_erp_FT{:}); 
% %% Baseline Correction
% subject_erp_bl = subject_erp;
% baselineindx = 200:900; % (-800 to -100)
% for subji = 1:size(subject_erp_bl,1)
%     for ki = 1:K
%         curdata = subject_erp_bl{subji,1}(:,ki);
%         blmean = nanmean(curdata(baselineindx));
%         newdata = curdata - blmean;
%         subject_erp_bl{subji,1}(:,ki) = newdata;
%     end
% end
% 
% %% BL corrected State ERPs
% state_erp_bl_GA = cell(K,1);
% for ki = 1:K
%     curstatedata = zeros(2000, size(hmm_data,1)); 
%     for subji = 1:size(hmm_data,1)
%         curstatedata(:,subji) = subject_erp_bl{subji,1}(:,ki);
%     end
%     
%     state_erp_bl_GA{ki,1} = mean(curstatedata,2); 
% end
% 
% numofstates = K;
% 
% numtoplot_r = floor(sqrt(numofstates));
% numtoplot_c = ceil(sqrt(numofstates));
% 
% figure;
% 
% for statei = 1:numofstates
%     subplot(numtoplot_r,numtoplot_c,statei)
%     curplot = plot(state_erp_bl_GA{statei,1}(200:1800),'LineWidth',2); 
%     curplot.XData = [-800:800];
%     vline(0, '--k');
%     grid on; 
%     title(['State:' num2str(statei)]); 
%     ylabel('Probability')
%     xlabel('Time (ms)')
%     xlim([-800 800]); 
% end
%% Getting Mean Data of States (only if zeromean = 0)

% p = parcellation('/Users/prabhjotdhami/Documents/MATLAB/Add-Ons/osl/parcellations/fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz');
% 
% % Do parcellation
% %D_epoched_parc = ROInets.get_node_tcs(D_epoched,p.parcelflag,'spatialBasis','Giles')
% 
% %D_epoched_parc = D_epoched_parc.montage('switch',0);
% 
% % We see states with visual, motor, frontal activations, etc
% %p = parcellation(atlasfile); % load the parcellation
% net_mean = getMean(hmm); % get the mean activity of each state
% net_mean = zscore(net_mean); % show activations for each state relative to the state average
% %p.osleyes(net_mean); % call osleyes
% 
% fig = p.plot_surface(net_mean)
% colormap(jet) 

%% Factorising the states spectra into frequency bands

X = hmm_data;
X = cat(1, X{:});

Tcat = cat(1, T{:});

fit = hmmspectramt(X,Tcat,Gamma,options); 

%[psd_tf,coh_tf,pdc_tf] = hmmtimefreq(fit,Gamma);

bands = [4 7; 8 12; 13 30; 31 45];
sp_fit = spectbands(fit,bands);

%% Plotting Frequency Maps
k = 1:K;
freqindex = 4;
parcellationfile = ('/Users/prabhjotdhami/Documents/MATLAB/Add-Ons/osl/parcellations/dk_cortical.nii.gz');
maskfile = ('/Users/prabhjotdhami/Documents/MATLAB/Add-Ons/osl/std_masks/MNI152_T1_8mm_brain.nii.gz');
centermaps = 1;
scalemaps = 0;
centergraphs = 0;
scalegraphs = 0;
outputfile = ['/Users/prabhjotdhami/Desktop/HMM/Results/sp_lmc_DK/TDE_map_lag29_freqbin' num2str(freqindex)];
wbdir = '/Applications/workbench/bin_macosx64';
type = 1; % 1 = coherence, 2 = partial coherence
threshold = 0.9;
labels = parcellation(parcellationfile);
labels = labels.labels;

maps = makeSpectralMap(sp_fit,k,freqindex,parcellationfile,maskfile,centermaps,scalemaps,outputfile,wbdir);
%graphs = makeSpectralGraph(sp_fit,k,freqindex,type,parcellationfile,maskfile,centergraphs,scalegraphs,threshold,outputfile);
%graphs = makeSpectralConnectivityCircle(sp_fit,k,freqindex,type,labels,centergraphs,scalegraphs,threshold);

%% Plotting maps
m = nii.load([outputfile '.nii.gz']);
p = parcellation(m);
p.plot_surface(m);
cmp = getPyPlot_cMap('RdYlBu_r'); 
%cmp(64:65,:) = [0.7098    0.7098    0.7098; 0.7098    0.7098    0.7098]; 
colormap(cmp)

%% Following Code Adopted from https://github.com/OHBA-analysis/HMM-MAR/blob/master/examples/NatComms2018_fullpipeline.m
%% Compute the spectra, at the group level and per subject
Hz = 1000; 
N = size(bst_source_data,2);
options_mt = struct('Fs',Hz); % Sampling rate - for the 25subj it is 300
options_mt.fpass = [4 45];  % band of frequency you're interested in
options_mt.tapers = [4 7]; % taper specification - leave it with default values
options_mt.p = 0; %0.01; % interval of confidence  
options_mt.win = 2 * Hz; % multitaper window
options_mt.to_do = [1 0]; % turn off pdc
options_mt.order = 0;
options_mt.embeddedlags = -29:29;

% average
fitmt = hmmspectramt(X,Tcat,Gamma,options_mt);

% per subject
fitmt_subj = cell(N,1);
d = length(options_mt.embeddedlags) - 1; 
acc = 0; 
for n=1:N
    X_subj = hmm_data{n};
    gamma = Gamma(acc + (1:(sum(T{n})-length(T{n})*d)),:);
    acc = acc + size(gamma,1);
    fitmt_subj{n} = hmmspectramt(X_subj,T{n},gamma,options_mt);
    fitmt_subj{n}.state = rmfield(fitmt_subj{n}.state,'ipsd');
    fitmt_subj{n}.state = rmfield(fitmt_subj{n}.state,'pcoh');
    fitmt_subj{n}.state = rmfield(fitmt_subj{n}.state,'phase');
    disp(['Subject ' num2str(n)])
end

%%

% Get the three bands depicted in the paper (the 4th is essentially capturing noise)
options_fact = struct();
options_fact.Ncomp = 5; 
options_fact.Base = 'coh';
[fitmt_group_fact_4b,sp_profiles_4b,fitmt_subj_fact_4b] = spectdecompose(fitmt_subj,options_fact);

% Get the wideband maps (the second is capturing noise)
options_fact.Ncomp = 2; 
[fitmt_group_fact_wb,sp_profiles_wb,fitmt_subj_fact_wb] = spectdecompose(fitmt_subj,options_fact);

% check if the spectral profiles make sense, if not you might like to repeat
figure; 
subplot(1,2,1); plot(sp_profiles_4b,'LineWidth',2)
subplot(1,2,2); plot(sp_profiles_wb,'LineWidth',2)

%% Do statistical testing on the spectral information

fitmt_subj_fact_1d = cell(N,1);
for n = 1:N
    fitmt_subj_fact_1d{n} = struct();
    fitmt_subj_fact_1d{n}.state = struct();
    for k = 1:K % we don't care about the second component
        fitmt_subj_fact_1d{n}.state(k).psd = fitmt_subj_fact_wb{n}.state(k).psd(1,:,:);
        fitmt_subj_fact_1d{n}.state(k).coh = fitmt_subj_fact_wb{n}.state(k).coh(1,:,:);
    end
end
tests_spectra = specttest(fitmt_subj_fact_1d,5000,1,1);
significant_spectra = spectsignificance(tests_spectra,0.01);

%% Making Maps
k = 1:K;
parcellationfile = ('/Users/prabhjotdhami/Documents/MATLAB/Add-Ons/osl/parcellations/dk_cortical.nii.gz');
maskfile = ('/Users/prabhjotdhami/Documents/MATLAB/Add-Ons/osl/std_masks/MNI152_T1_8mm_brain.nii.gz');
centermaps = 1;
scalemaps = 0;
outputfile = ['/Users/prabhjotdhami/Desktop/HMM/Results/sp_lmc_DK/'];
wbdir = '/Applications/workbench/bin_macosx64';

% Wideband
mapfile = [outputfile '/state_maps_wideband'];
maps = makeSpectralMap(fitmt_group_fact_wb,k,1,parcellationfile,maskfile,centermaps,scalemaps,mapfile,wbdir);

% Per frequency band
for fr = 1:4
    mapfile = [outputfile '/state_maps_band' num2str(fr)];
    maps = makeSpectralMap(fitmt_group_fact_4b,k,fr,parcellationfile,maskfile,centermaps,scalemaps,mapfile,wbdir);
end

%% Plotting maps
m = nii.load(['state_maps_band4.nii.gz']);
p = parcellation(m);
p.plot_surface(m);
cmp = getPyPlot_cMap('RdYlBu_r'); 
colormap(cmp)

%% Get glass connectivity brains
% doing data-driven thresholding

parcfile = ('/Users/prabhjotdhami/Documents/MATLAB/Add-Ons/osl/parcellations/dk_cortical.nii.gz');

K = length(fitmt_group_fact_4b.state); ndim = 68; 
spatialRes = 8; edgeLims = [4 8];

% wideband
M = zeros(ndim); 
for k = 1:K
    M = M + squeeze(abs(fitmt_group_fact_wb.state(k).coh(1,:,:))) / K;
end
for k = 1:K
    graph = squeeze(abs(fitmt_group_fact_wb.state(k).coh(1,:,:)));
    graph = (graph - M);  
    tmp = triu(graph); tmp = tmp(:);
    inds2 = find(tmp>1e-10);
    data = tmp(inds2);
    S2 = [];
    S2.data = data;
    S2.do_fischer_xform = false;
    S2.do_plots = 1;
    S2.pvalue_th = 0.01/length(S2.data);
    graph_ggm = teh_graph_gmm_fit(S2); 
    th = graph_ggm.normalised_th;
    graph = graph_ggm.data';
    graph(graph<th) = NaN;
    graphmat = nan(ndim, ndim);
    graphmat(inds2) = graph;
    graph = graphmat;
    p = parcellation(parcfile);
    spatialMap = p.to_matrix(p.weight_mask);
    % compensate the parcels to have comparable weights
    for j=1:size(spatialMap,2) % iterate through regions : make max value to be 1
        spatialMap(:,j) =  spatialMap(:,j) / max(spatialMap(:,j));
    end
    p.weight_mask = p.to_vol(spatialMap);
    [h_patch,h_scatter] = p.plot_network(graph,th);
end
%% Per frequency band
for fr = 1:4
    M = zeros(ndim);
    for k = 1:K
        M = M + squeeze(abs(fitmt_group_fact_4b.state(k).coh(fr,:,:))) / K;
    end
    for k = 1:K
        graph = squeeze(abs(fitmt_group_fact_4b.state(k).coh(fr,:,:)));
        graph = (graph - M);
        tmp = triu(graph); tmp = tmp(:);
        inds2 = find(tmp>1e-10);
        data = tmp(inds2);
        S2 = [];
        S2.data = data;
        S2.do_fischer_xform = false;
        S2.do_plots = 1;
        S2.pvalue_th = 0.01/length(S2.data);
        graph_ggm = teh_graph_gmm_fit(S2);
        th = graph_ggm.normalised_th;
        graph = graph_ggm.data';
        graph(graph<th) = NaN;
        graphmat = nan(ndim, ndim);
        graphmat(inds2) = graph;
        graph = graphmat;
        p = parcellation(parcfile);
        spatialMap = p.to_matrix(p.weight_mask);
        % compensate the parcels to have comparable weights
        for j=1:size(spatialMap,2) % iterate through regions : make max value to be 1
            spatialMap(:,j) =  spatialMap(:,j) / max(spatialMap(:,j));
        end
        p.weight_mask = p.to_vol(spatialMap);
        [h_patch,h_scatter] = p.plot_network(graph,th);
    end
end
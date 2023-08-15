function hmm = hmmhsinit(hmm,GammaInit,T)
% Initialise variables related to the Markov chain
%
% hmm		hmm data structure
%
% OUTPUT
% hmm           hmm structure
%
% Author: Diego Vidaurre, OHBA, University of Oxford

% if isfield(hmm.train,'grouping')
%     Q = length(unique(hmm.train.grouping));
% else
%     Q = 1;
% end
Q = 1;

if hmm.train.episodic
    
    % define P-priors
    defhmmprior = struct('Dir2d_alpha',[],'Dir_alpha',[]);
    defhmmprior.Dir_alpha = [0 1];
    defhmmprior.Dir2d_alpha = ones(2);
    defhmmprior.Dir2d_alpha(eye(2)==1) = hmm.train.DirichletDiag; % effect on diagonal
    if hmm.train.ehmm_priorOFFvsON > 1 % effect on ->OFF
        defhmmprior.Dir2d_alpha(:,2) = defhmmprior.Dir2d_alpha(:,2) * hmm.train.ehmm_priorOFFvsON; 
    else % effect on ->ON
        defhmmprior.Dir2d_alpha(:,1) = defhmmprior.Dir2d_alpha(:,1) * (1/hmm.train.ehmm_priorOFFvsON); 
    end
    defhmmprior.Dir2d_alpha = defhmmprior.Dir2d_alpha * hmm.train.PriorWeightingP; % prior's weight
    %%%defhmmprior.Dir2d_alpha = [1000 10; 10 1000];
    for k = 1:hmm.K
        hmm.state(k).prior = defhmmprior;
    end
    % assigning initial posteriors and priors for hidden state
    if nargin > 1 && ~isempty(GammaInit) && hmm.train.updateP
        hmm = hsupdate_ehmm([],GammaInit,T,hmm);
    elseif ~hmm.train.updateP && (~isfield(hmm.state(1),'P') || isempty(hmm.state(1).P))
        for k = 1:hmm.K
            Dir2d_alpha_norm = defhmmprior.Dir2d_alpha;
            Dir2d_alpha_norm = Dir2d_alpha_norm / sum(Dir2d_alpha_norm(:));
            hmm.state(k).Dir2d_alpha = T * Dir2d_alpha_norm;
            hmm.state(k).P = zeros(2);
            for k2 = 1:2
                hmm.state(k).P(k2,:) = hmm.state(k).Dir2d_alpha(k2,:) ./ ...
                    sum(hmm.state(k).Dir2d_alpha(k2,:));
            end
        end
    else % this is reached basically if it's a warm restart
        % do nothing
    end
    % Initial state
    for k = 1:hmm.K
        hmm.state(k).Dir_alpha = [0 1];
        hmm.state(k).Pi = [0 1];
    end

else
    
    % define P-priors
    defhmmprior = struct('Dir2d_alpha',[],'Dir_alpha',[]);
    defhmmprior.Dir_alpha = hmm.train.PriorWeightingPi * ones(1,hmm.K);
    defhmmprior.Dir_alpha(~hmm.train.Pistructure) = 0;
    if isfield(hmm.train,'cluster') && hmm.train.cluster
        defhmmprior.Dir2d_alpha = eye(hmm.K);
    else
        defhmmprior.Dir2d_alpha = ones(hmm.K);
        defhmmprior.Dir2d_alpha(eye(hmm.K)==1) = hmm.train.DirichletDiag;
        defhmmprior.Dir2d_alpha(~hmm.train.Pstructure) = 0;
        defhmmprior.Dir2d_alpha = hmm.train.PriorWeightingP .* defhmmprior.Dir2d_alpha;
    end
    % assigning default priors for hidden states
    if ~isfield(hmm,'prior')
        hmm.prior = defhmmprior;
    else
        % priors not specified are set to default
        hmmpriorlist = fieldnames(defhmmprior);
        fldname = fieldnames(hmm.prior);
        misfldname = find(~ismember(hmmpriorlist,fldname));
        for i = 1:length(misfldname)
            priorval = getfield(defhmmprior,hmmpriorlist{i});
            hmm.prior = setfield(hmm.prior,hmmpriorlist{i},priorval);
        end
    end
    
    if nargin > 1 && ~isempty(GammaInit) && hmm.train.updateP
        hmm = hsupdate([],GammaInit,T,hmm);
    else
        % Initial state
        kk = hmm.train.Pistructure;
        if Q==1
            hmm.Dir_alpha = zeros(1,hmm.K);
            hmm.Dir_alpha(kk) = hmm.train.PriorWeightingPi;
            hmm.Pi = zeros(1,hmm.K);
            hmm.Pi(kk) = ones(1,sum(kk)) / sum(kk);
        else
            hmm.Dir_alpha = zeros(hmm.K,Q);
            hmm.Dir_alpha(kk,:) = 1;
            hmm.Pi = zeros(hmm.K,Q);
            hmm.Pi(kk,:) = ones(sum(kk),Q) / sum(kk);
        end
        % State transitions
        if hmm.train.cluster
            hmm.Dir2d_alpha = eye(hmm.K);
            hmm.P = eye(hmm.K);
        else
            hmm.Dir2d_alpha = zeros(hmm.K,hmm.K,Q);
            hmm.P = zeros(hmm.K,hmm.K,Q);
            for i = 1:Q
                for k = 1:hmm.K
                    kk = (hmm.train.Pstructure(k,:)==1);
                    hmm.Dir2d_alpha(k,kk,i) = 1;
                    if length(hmm.train.DirichletDiag) == 1
                        hmm.Dir2d_alpha(k,k,i) = hmm.train.DirichletDiag;
                    else
                        hmm.Dir2d_alpha(k,k,i) = hmm.train.DirichletDiag(k);
                    end
                    hmm.Dir2d_alpha(k,kk,i) = hmm.train.PriorWeightingP .* hmm.Dir2d_alpha(k,kk,i);
                    hmm.P(k,kk,i) = hmm.Dir2d_alpha(k,kk,i) ./ sum(hmm.Dir2d_alpha(k,kk,i));
                end
            end
        end
    end
    
end

end

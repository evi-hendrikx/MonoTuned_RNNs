% close all
clearvars -except ve_data

minVE = 0.2;
minDuration = 0.06;
maxDuration = 0.99;
minPeriod = 0.06;
maxPeriod =0.99;


%% load and prepare data
% load('parameters_visual_field_maps.mat')

%choose which models you want to use
modelFieldNames = fieldnames(ve_data);
use_models = modelFieldNames(1); % fill in models you want
cv_data = monoTuned_cv_timing_data_anonymous(ve_data, use_models);
modelFieldNames = fieldnames(cv_data);
subjNames = fieldnames(cv_data.(modelFieldNames{1}));
condNames = fieldnames(cv_data.(modelFieldNames{1}).(subjNames{1}));
ROILabels = fieldnames(cv_data.(modelFieldNames{1}).(subjNames{1}).(condNames{1}).crossValidated); %possible, because S3 has all maps
ROIs = unique(erase(ROILabels, ["Right","Left","right","left"]),'stable');


%% prepare data for ANOVA
hemispheres = {'left','right'};
meanSub = [];

% get mean values per roi per run, put in struct with runs and hemispheres
% in same column
for subj = 1:length(subjNames)
    for roi = 1:length(ROILabels)

        if roi <= length(ROIs) % left
            roi_id = roi;
            hemisphere = 1;
        else
            roi_id = roi - length(ROIs); % right
            hemisphere = 2;
        end

        roi_names_subj = fieldnames(cv_data.(modelFieldNames{1}).(subjNames{subj}).(condNames{1}).crossValidated);
        for condition = 1:length(condNames)

            if any(strcmp(roi_names_subj, ROILabels{roi}))

                %NOTE: if tuned model is involved, it's always the second model (due to build cv data)
                % variance explained on not cross-validated % necessary for voxel selection
                ve_mod1_Current_pre_cv = cv_data.(modelFieldNames{1}).(subjNames{subj}).(char(condNames{condition})).used_for_crossValidation.(ROILabels{roi}).varianceExplained';

                % compressive exponents (and if second model is tuned preferred duration/period)
                xs_mod1_Current = cv_data.(modelFieldNames{1}).(subjNames{subj}).(char(condNames{condition})).crossValidated.(ROILabels{roi}).xs';
                ys_mod1_Current = cv_data.(modelFieldNames{1}).(subjNames{subj}).(char(condNames{condition})).crossValidated.(ROILabels{roi}).ys';

                % beta mono
                beta_xs_mod1_Current = cv_data.(modelFieldNames{1}).(subjNames{subj}).(char(condNames{condition})).used_for_crossValidation.(ROILabels{roi}).beta1;
                beta_ys_mod1_Current = cv_data.(modelFieldNames{1}).(subjNames{subj}).(char(condNames{condition})).used_for_crossValidation.(ROILabels{roi}).beta2;

                % NOTE: different model selection than in the other analyses. Here we
                % want voxels selected for model 1 that do well for model 1 (>minVE),
                % and select voxels for model 2 that do well for model 2
                % (>minVE) separately
                % voxel selection at subject level, remove if < minVE for uncrossvalidated data
                removeVE_mod1 = find(ve_mod1_Current_pre_cv<=minVE);
                xs_mod1_Current(removeVE_mod1) = [];
                ys_mod1_Current(removeVE_mod1) = [];

                % these (the "beta" parameters) have a slightly different voxel selection and
                % therefore sometimes iCrds is slightly not the same size
                % as the "beta" values
                rmCrds1 = cv_data.(modelFieldNames{1}).(subjNames{subj}).(char(condNames{condition})).crossValidated.(ROILabels{roi}).iCoords(removeVE_mod1);
                betaCrds = cv_data.(modelFieldNames{1}).(subjNames{subj}).(char(condNames{condition})).crossValidated.(ROILabels{roi}).beta_Coords;
                Crds = cv_data.(modelFieldNames{1}).(subjNames{subj}).(char(condNames{condition})).crossValidated.(ROILabels{roi}).iCoords;

                if length(Crds) ~= length(betaCrds)
                    disp(subj)
                end
                [overlappingCrds,CrdsOrder,betaOrder]=intersect(Crds, betaCrds);
                beta_xs_mod1_Current = beta_xs_mod1_Current(betaOrder,:);
                beta_ys_mod1_Current = beta_ys_mod1_Current(betaOrder,:);

                [~,removeVE_mod1_betaSelection,~] = intersect(CrdsOrder,removeVE_mod1);
                beta_xs_mod1_Current(removeVE_mod1_betaSelection,:) = [];
                beta_ys_mod1_Current(removeVE_mod1_betaSelection,:) = [];

                overlappingCrds(removeVE_mod1_betaSelection) = [];
                Crds_remove_selection = Crds;
                Crds_remove_selection(removeVE_mod1) = [];
                [~,beta_overlap_of_already_reduced,~] = intersect(Crds_remove_selection,overlappingCrds);
                
                % check from here
                beta_ratio_mod1_Current = beta_xs_mod1_Current./beta_ys_mod1_Current;
                beta_order_xs = xs_mod1_Current(beta_overlap_of_already_reduced);
                beta_order_ys = ys_mod1_Current(beta_overlap_of_already_reduced);
                beta_ratio_mod1_Current(beta_order_ys==1) = Inf;
                beta_ratio_mod1_Current(beta_order_xs==0) = 0;
                beta_order_xs(nanmedian(beta_ratio_mod1_Current,2)==0) = 0;
                beta_order_ys(nanmedian(beta_ratio_mod1_Current,2)==Inf) = 1;
                xs_mod1_Current(beta_overlap_of_already_reduced) = beta_order_xs;
                ys_mod1_Current(beta_overlap_of_already_reduced) = beta_order_ys;

%                 beta_ratio_mod1_Current(beta_ratio_mod1_Current==0) = NaN;
%                 beta_ratio_mod1_Current(beta_ratio_mod1_Current==Inf) = NaN;
                
                beta_ratio_mod1_Current = nanmedian(beta_ratio_mod1_Current,2);


            else
                xs_mod1_Current = [];
                ys_mod1_Current = [];
                xs_mod2_Current = [];
                ys_mod2_Current = [];
                beta_xs_mod1_Current = [];
                beta_ys_mod1_Current = [];
                beta_ratio_mod1_Current = [];
            end


            meanSub.xs.(modelFieldNames{1})(subj,hemisphere,condition,roi_id)=nanmedian(xs_mod1_Current);
            meanSub.ys.(modelFieldNames{1})(subj,hemisphere,condition,roi_id)=nanmedian(ys_mod1_Current);

            meanSub.beta_xs.(modelFieldNames{1})(subj,hemisphere,condition,roi_id)=nanmedian(mean(beta_xs_mod1_Current,2));% so also works for S3 and S2
            meanSub.beta_ys.(modelFieldNames{1})(subj,hemisphere,condition,roi_id)=nanmedian(mean(beta_ys_mod1_Current,2));% so also works for S3 and S2
            meanSub.beta_ratio.(modelFieldNames{1})(subj,hemisphere,condition,roi_id)=nanmedian(beta_ratio_mod1_Current);
            meanSubSubj(subj,hemisphere,condition,roi_id) = subjNames(subj);
            meanSubRoi(subj,hemisphere,condition,roi_id) = ROIs(roi_id);
            meanSubHemi(subj,hemisphere,condition,roi_id) = hemispheres(hemisphere);
            meanSubCond(subj,hemisphere,condition,roi_id) = condNames(condition);
        end

    end
end

meanSubXs = meanSub.xs.MonoOcc;
meanSubYs = meanSub.ys.MonoOcc;
meanSubRatio = meanSub.beta_ratio.MonoOcc;

#include <vector>

void compute_likelihoods_emu(){


	TFile *f_pdf = new TFile("likelihood_histograms/pdf_beamlike_emu_500bins_root5.root","READ");
	TH1F *pdf_mu_charge = (TH1F*) f_pdf->Get("pdf_beamlike_muon_charge");
	TH1F *pdf_e_charge = (TH1F*) f_pdf->Get("pdf_beamlike_electron_charge");
	TH1F *pdf_mu_time = (TH1F*) f_pdf->Get("pdf_beamlike_muon_time");
	TH1F *pdf_e_time = (TH1F*) f_pdf->Get("pdf_beamlike_electron_time");
	TH1F *pdf_mu_theta = (TH1F*) f_pdf->Get("pdf_beamlike_muon_thetaB");
	TH1F *pdf_e_theta = (TH1F*) f_pdf->Get("pdf_beamlike_electron_thetaB");
	TH1F *pdf_mu_phi = (TH1F*) f_pdf->Get("pdf_beamlike_muon_phiB");
	TH1F *pdf_e_phi = (TH1F*) f_pdf->Get("pdf_beamlike_electron_phiB");

	//Vector of pdfs
	std::vector<TH1F*> pdf_hists_electron = {pdf_e_charge, pdf_e_time, pdf_e_theta, pdf_e_phi};
	std::vector<TH1F*> pdf_hists_muon = {pdf_mu_charge, pdf_mu_time, pdf_mu_theta, pdf_mu_phi};

	//Output ROOT file
	TFile *out = new TFile("likelihood_result_emu_500bins_root5.root","RECREATE");
	int n_bins = 500;

	// Test histograms (drawn event distributions, likelihood value results)
	TH1F *test_muon_charge = new TH1F("test_muon_charge","Test distribution Q muon",n_bins,10,100);
	TH1F *test_electron_charge = new TH1F("test_electron_charge","Test distribution Q electron",n_bins,10,100);
	TH1F *test_muon_time = new TH1F("test_muon_time","Test distribution T muon",n_bins,0,20);
	TH1F *test_electron_time = new TH1F("test_electron_time","Test distribution T electron",n_bins,0,20);
	TH1F *test_muon_theta = new TH1F("test_muon_theta","Test distribution Theta muon",n_bins,-1.2,2.2);
	TH1F *test_electron_theta = new TH1F("test_electron_theta","Test distribution Theta electron",n_bins,-1.2,2.2);
	TH1F *test_muon_phi = new TH1F("test_muon_phi","Test distribution Phi muon",n_bins,-3.2,3.2);
	TH1F *test_electron_phi = new TH1F("test_electron_phi","Test distribution Phi electron",n_bins,-3.2,3.2);
	TH1F *likelihood_test_muon_charge = new TH1F("likelihood_test_muon_charge","Likelihood_test_muon_charge",200,-5,5);
	TH1F *likelihood_test_electron_charge = new TH1F ("likelihood_test_electron_charge","Likelihood_test_electron_charge",200,-5,5);
	TH1F *likelihood_test_muon_time = new TH1F("likelihood_test_muon_time","Likelihood_test_muon_time",200,-5,5);
	TH1F *likelihood_test_electron_time = new TH1F ("likelihood_test_electron_time","Likelihood_test_electron_time",200,-5,5);
	TH1F *likelihood_test_muon_theta = new TH1F("likelihood_test_muon_charge","Likelihood_test_muon_theta",200,-5,5);
	TH1F *likelihood_test_electron_theta = new TH1F ("likelihood_test_electron_charge","Likelihood_test_electron_theta",200,-5,5);
	TH1F *likelihood_test_muon_phi = new TH1F("likelihood_test_muon_phi","Likelihood_test_muon_phi",200,-5,5);
	TH1F *likelihood_test_electron_phi = new TH1F ("likelihood_test_electron_phi","Likelihood_test_electron_phi",200,-5,5);

	//Vector of likelihood & event histograms
	std::vector<TH1F*> test_hists_muon = {test_muon_charge, test_muon_time, test_muon_theta, test_muon_phi};
	std::vector<TH1F*> test_hists_electron = {test_electron_charge, test_electron_time, test_electron_theta, test_electron_phi};
	std::vector<TH1F*> likelihood_test_hists_muon = {likelihood_test_muon_charge, likelihood_test_muon_time, likelihood_test_muon_theta, likelihood_test_muon_phi};
	std::vector<TH1F*> likelihood_test_hists_electron = {likelihood_test_electron_charge, likelihood_test_electron_time, likelihood_test_electron_theta, likelihood_test_electron_phi};

	// Test theoretical performance of likelihood values by drawing events from the pdfs
	const int n_rebin = 8;
	Int_t rebin[n_rebin]={1,2,5,10,20,25,50,100};
	std::vector<std::string> var_names = {"charge","time","theta","phi"};
	int n_toy_events = 10000;
	int nhits_per_event = 20;

	for (int i_rebin = 0; i_rebin < n_rebin; i_rebin++) {

		for (int i_hist=0; i_hist < (int) pdf_hists_electron.size(); i_hist++){

			TH1F *lklh_muon = (TH1F*) likelihood_test_hists_muon.at(i_hist)->Clone();
			TH1F *lklh_electron = (TH1F*) likelihood_test_hists_electron.at(i_hist)->Clone();
			lklh_muon->Reset();
			lklh_electron->Reset();
			std::stringstream histname_muon, histname_electron;
			histname_muon << "likelihood_test_muon_"<<var_names[i_hist]<<"_rebin"<<rebin[i_rebin];
			histname_electron << "likelihood_test_electron_"<<var_names[i_hist]<<"_rebin"<<rebin[i_rebin];
			lklh_muon->SetName(histname_muon.str().c_str());
			lklh_electron->SetName(histname_electron.str().c_str());

			TH1F *temp_pdf_muon_rebin = (TH1F*) ((TH1F*) pdf_hists_muon.at(i_hist)->Clone())->Rebin(rebin[i_rebin]);
			TH1F *temp_pdf_electron_rebin = (TH1F*) ((TH1F*) pdf_hists_electron.at(i_hist)->Clone())->Rebin(rebin[i_rebin]);
			TH1F *temp_test_muon_rebin = (TH1F*) ((TH1F*) test_hists_muon.at(i_hist)->Clone())->Rebin(rebin[i_rebin]);
			TH1F *temp_test_electron_rebin = (TH1F*) ((TH1F*) test_hists_electron.at(i_hist)->Clone())->Rebin(rebin[i_rebin]);
			
			for  (int i_toy=0; i_toy < n_toy_events; i_toy++){
				temp_test_muon_rebin->Reset();
				temp_test_electron_rebin->Reset();
				for (int i_hit=0; i_hit < nhits_per_event; i_hit++){
					temp_test_muon_rebin->Fill(temp_pdf_muon_rebin->GetRandom());
					temp_test_electron_rebin->Fill(temp_pdf_electron_rebin->GetRandom());
				}

				double chi2_Muon_muon = temp_pdf_muon_rebin->Chi2Test(temp_test_muon_rebin,"UUNORMCHI2/NDF");
				double chi2_Muon_electron = temp_pdf_electron_rebin->Chi2Test(temp_test_muon_rebin,"UUNORMCHI2/NDF");
				double chi2_Electron_muon = temp_pdf_muon_rebin->Chi2Test(temp_test_electron_rebin,"UUNORMCHI2/NDF");
				double chi2_Electron_electron = temp_pdf_electron_rebin->Chi2Test(temp_test_electron_rebin,"UUNORMCHI2/NDF");

				lklh_muon->Fill(chi2_Muon_electron-chi2_Muon_muon);
				lklh_electron->Fill(chi2_Electron_electron-chi2_Electron_muon);

			}

			lklh_muon->GetXaxis()->SetTitle("#Delta #chi^{2} = #chi^{2}_{electron}-#chi^{2}_{muon}");
			lklh_electron->GetXaxis()->SetTitle("#Delta #chi^{2} = #chi^{2}_{electron}-#chi^{2}_{muon}");
			lklh_muon->SetLineColor(2);
			lklh_muon->SetFillColor(kRed);
			lklh_muon->SetFillStyle(3005);
			lklh_electron->SetLineColor(4);
			lklh_electron->SetFillColor(kBlue);
			lklh_electron->SetFillStyle(3013);

			out->cd();
			lklh_muon->Write();
			lklh_electron->Write();

			std::stringstream canvas_name_test;
			canvas_name_test << "canvas_test_"<<var_names[i_hist]<<"rebin"<<rebin[i_rebin];
			TLegend *leg_test = new TLegend(0.75,0.75,0.9,0.9);
			TCanvas *c_test = new TCanvas(canvas_name_test.str().c_str(),canvas_name_test.str().c_str(),900,600);
			c_test->cd();
			lklh_muon->SetStats(0);
			std::stringstream hist_title;
			hist_title <<"PID Likelihood "<<var_names[i_hist]<<" ("<<n_bins/rebin[i_rebin]<<" bins)";
			lklh_muon->SetTitle(hist_title.str().c_str());
			lklh_muon->Draw("HIST");
			lklh_electron->Draw("same HIST");
			leg_test->AddEntry(lklh_muon,"muon","l");
			leg_test->AddEntry(lklh_electron,"electron","l");
			leg_test->Draw();
			c_test->Write();

			if (i_rebin == 0 && i_hist == 0) c_test->Print("PID_Likelihood_Test.pdf(","pdf");
			else if (i_rebin == n_rebin-1 && i_hist == (int) pdf_hists_electron.size()-1) c_test->Print("PID_Likelihood_Test.pdf)","pdf");
			else c_test->Print("PID_Likelihood_Test.pdf","pdf");

		}

	}

	//TFile *f_data_e = new TFile("data/beamlike_electrons_histogram_DigitThr10_0_999_nolklh.root","READ");
	TFile *f_data_rings = new TFile("data/beam_DigitThr10_0_4996.root","READ");


	TTree *classification_e = (TTree*) f_data_rings->Get("classification_tree");
	int nentries = classification_e->GetEntries();
	std::vector<double>* event_charge = new std::vector<double>;
	std::vector<double>* event_time = new std::vector<double>;
	std::vector<double>* event_theta = new std::vector<double>;
	std::vector<double>* event_phi = new std::vector<double>;

	classification_e->SetBranchAddress("PMTQVector",&event_charge);
	classification_e->SetBranchAddress("PMTTVector",&event_time);
	classification_e->SetBranchAddress("PMTThetaBaryVector",&event_theta);
	classification_e->SetBranchAddress("PMTPhiBaryVector",&event_phi);	

	//Add newly calculated classification variables
	double lklh_charge, lklh_time, lklh_theta, lklh_phi;
	std::vector<double> lklh_values = {lklh_charge, lklh_time, lklh_theta, lklh_phi};
	/*TBranch *b_charge = classification_e->Branch("PMTLikelihoodQ",&lklh_values.at(0));
	TBranch *b_time = classification_e->Branch("PMTLikelihoodT",&lklh_values.at(1));
	TBranch *b_theta = classification_e->Branch("PMTLikelihoodTheta",&lklh_values.at(2));
	TBranch *b_phi = classification_e->Branch("PMTLikelihoodPhi",&lklh_values.at(3));*/

	TH1F *hist_PMTLikelihoodQ = new TH1F("hist_PMTLikelihoodQ","hist_PMTLikelihoodQ",200,-5,5);
	TH1F *hist_PMTLikelihoodT = new TH1F("hist_PMTLikelihoodT","hist_PMTLikelihoodT",200,-5,5);
	TH1F *hist_PMTLikelihoodTheta = new TH1F("hist_PMTLikelihoodTheta","hist_PMTLikelihoodTheta",200,-5,5);
	TH1F *hist_PMTLikelihoodPhi = new TH1F("hist_PMTLikelihoodPhi","hist_PMTLikelihoodPhi",200,-5,5);

	//Read-in muon file
	TFile *f_data_mu = new TFile("data/beamlike_muons_histogram_DigitThr10_0_399_nolklh.root","READ");

	TTree *classification_mu = (TTree*) f_data_mu->Get("classification_tree");
	int nentries_mu = classification_mu->GetEntries();
	std::vector<double>* event_charge_mu = new std::vector<double>;
	std::vector<double>* event_time_mu = new std::vector<double>;
	std::vector<double>* event_theta_mu = new std::vector<double>;
	std::vector<double>* event_phi_mu = new std::vector<double>;
	classification_mu->SetBranchAddress("PMTQVector",&event_charge_mu);
	classification_mu->SetBranchAddress("PMTTVector",&event_time_mu);
	classification_mu->SetBranchAddress("PMTThetaBaryVector",&event_theta_mu);
	classification_mu->SetBranchAddress("PMTPhiBaryVector",&event_phi_mu);

	//Add newly calculated classification variables
	double lklh_charge_mu, lklh_time_mu, lklh_theta_mu, lklh_phi_mu;
	std::vector<double> lklh_values_mu = {lklh_charge_mu, lklh_time_mu, lklh_theta_mu, lklh_phi_mu};
	/*TBranch *b_charge_mu = classification_mu->Branch("PMTLikelihoodQ",&lklh_values_mu.at(0));
	TBranch *b_time_mu = classification_mu->Branch("PMTLikelihoodT",&lklh_values_mu.at(1));
	TBranch *b_theta_mu = classification_mu->Branch("PMTLikelihoodTheta",&lklh_values_mu.at(2));
	TBranch *b_phi_mu = classification_mu->Branch("PMTLikelihoodPhi",&lklh_values_mu.at(3));
*/
	// Add likelihood comparison histograms
	TH1F *likelihood_e_charge = new TH1F("likelihood_e_charge","Likelihood Q e",200,-5,5);
	TH1F *likelihood_mu_charge = new TH1F("likelihood_mu_charge","Likelihood Q mu",200,-5,5);
	TH1F *likelihood_e_time = new TH1F("likelihood_e_time","Likelihood T e",200,-5,5);
	TH1F *likelihood_mu_time = new TH1F("likelihood_mu_time","Likelihood T mu",200,-5,5);
	TH1F *likelihood_e_theta = new TH1F("likelihood_e_theta","Likelihood Theta e",200,-5,5);
	TH1F *likelihood_mu_theta = new TH1F("likelihood_mu_theta","Likelihood Theta mu",200,-5,5);
	TH1F *likelihood_e_phi = new TH1F("likelihood_e_phi","Likelihood Phi e",200,-5,5);
	TH1F *likelihood_mu_phi = new TH1F("likelihood_mu_phi","Likelihood Phi mu",200,-5,5);

	std::vector<TH1F*> likelihood_e = {likelihood_e_charge, likelihood_e_time, likelihood_e_theta, likelihood_e_phi};
	std::vector<TH1F*> likelihood_mu = {likelihood_mu_charge, likelihood_mu_time, likelihood_mu_theta, likelihood_mu_phi};

	TH1F *temp_hist_charge = new TH1F("temp_hist_charge","Temp hist",n_bins,10,100);
	TH1F *temp_hist_time = new TH1F("temp_hist_time","Temp hist t",n_bins,0,20);
	TH1F *temp_hist_theta = new TH1F("temp_hist_theta","Temp hist theta",n_bins,-1.2,2.2);
	TH1F *temp_hist_phi = new TH1F("temp_hist_phi","Temp hist phi",n_bins,-3.2,3.2);
	TH1F *num_pmts = new TH1F("num_pmts","Number of PMTs",130,0,130);

	std::vector<TH1F*> temp_hists = {temp_hist_charge, temp_hist_time, temp_hist_theta, temp_hist_phi};

	//ofstream file_electrons("beamlike_electrons_pdf.csv");
	//file_electrons << "LikelihoodQ,LikelihoodT,LikelihoodTheta,LikelihoodPhi"<<std::endl;
	//ofstream file_muons("beamlike_muons_pdf.csv");
	//file_muons << "LikelihoodQ,LikelihoodT,LikelihoodTheta,LikelihoodPhi"<<std::endl;
	
	ofstream file_rings("beam_emu_likelihood.csv");
	file_rings << "LikelihoodQ,LikelihoodT,LikelihoodTheta,LikelihoodPhi"<<std::endl;	

	ofstream file_overlap_emu("beamlike_emu_overlap.csv");
	file_overlap_emu << "Bins,OverlapQ,OverlapT,OverlapTheta,OverlapPhi"<<std::endl;
	std::vector<std::vector<double>> overlap_vector;


	std::vector<double> vec_lklh_charge_e;
	std::vector<double> vec_lklh_time_e;
	std::vector<double> vec_lklh_theta_e;
	std::vector<double> vec_lklh_phi_e;
	std::vector<double> vec_lklh_charge_mu;
	std::vector<double> vec_lklh_time_mu;
	std::vector<double> vec_lklh_theta_mu;
	std::vector<double> vec_lklh_phi_mu;

	std::vector<int> best_rebin_e = {50,1,1,1};
	std::vector<int> best_rebin_mu = {50,1,1,1};


	for (int i_rebin = 0; i_rebin < n_rebin; i_rebin++){

		std::vector<double> temp_overlap;

		file_overlap_emu << rebin[i_rebin] <<",";

		TH1F *temp_hist_charge_rebin = (TH1F*) ((TH1F*)temp_hist_charge->Clone())->Rebin(rebin[i_rebin]);
		TH1F *temp_hist_time_rebin = (TH1F*) ((TH1F*)temp_hist_time->Clone())->Rebin(rebin[i_rebin]);
		TH1F *temp_hist_theta_rebin = (TH1F*) ((TH1F*)temp_hist_theta->Clone())->Rebin(rebin[i_rebin]);
		TH1F *temp_hist_phi_rebin = (TH1F*) ((TH1F*)temp_hist_phi->Clone())->Rebin(rebin[i_rebin]);

		std::vector<TH1F*> temp_hist_rebin={temp_hist_charge_rebin,temp_hist_time_rebin,temp_hist_theta_rebin,temp_hist_phi_rebin};
		std::vector<TH1F*> likelihood_muon_rebin;
		std::vector<TH1F*> likelihood_electron_rebin;
		std::vector<TH1F*> pdf_muon_rebin;
		std::vector<TH1F*> pdf_electron_rebin;


		for (int i_hist=0; i_hist < (int) pdf_hists_muon.size(); i_hist++){
			TH1F *lklhd_muon = (TH1F*) likelihood_mu.at(i_hist)->Clone();
			TH1F *lklhd_electron = (TH1F*) likelihood_e.at(i_hist)->Clone();
			lklhd_muon->Reset();
			lklhd_electron->Reset();
			std::stringstream histname_muon, histname_electron;
			histname_muon << "likelihood_muon_"<<var_names[i_hist]<<"_rebin"<<rebin[i_rebin];
			histname_electron << "likelihood_electron_"<<var_names[i_hist]<<"_rebin"<<rebin[i_rebin];
			lklhd_muon->SetName(histname_muon.str().c_str());
			lklhd_electron->SetName(histname_electron.str().c_str());
			likelihood_muon_rebin.push_back(lklhd_muon);
			likelihood_electron_rebin.push_back(lklhd_electron);

			TH1F *temp_pdf_muon_rebin = (TH1F*) ((TH1F*) pdf_hists_muon.at(i_hist)->Clone())->Rebin(rebin[i_rebin]);
			TH1F *temp_pdf_electron_rebin = (TH1F*) ((TH1F*) pdf_hists_electron.at(i_hist)->Clone())->Rebin(rebin[i_rebin]);
			
			pdf_muon_rebin.push_back(temp_pdf_muon_rebin);
			pdf_electron_rebin.push_back(temp_pdf_electron_rebin);
		}

		// Loop over electrons-file

		for (int i_e=0; i_e < nentries; i_e++){
		
			temp_hist_charge_rebin->Reset();
			temp_hist_time_rebin->Reset();
			temp_hist_theta_rebin->Reset();
			temp_hist_phi_rebin->Reset();

			classification_e->GetEntry(i_e);

			for (int i_q = 0; i_q < (int) event_charge->size(); i_q++){
				temp_hist_charge_rebin->Fill(event_charge->at(i_q));
				temp_hist_time_rebin->Fill(event_time->at(i_q));
				temp_hist_theta_rebin->Fill(event_theta->at(i_q));
				temp_hist_phi_rebin->Fill(event_phi->at(i_q));
			}

			num_pmts->Fill(event_charge->size());

			for (int i_hist=0; i_hist< (int) pdf_hists_electron.size(); i_hist++){

				double chi2_muon = pdf_muon_rebin.at(i_hist)->Chi2Test(temp_hist_rebin.at(i_hist),"UUNORMCHI2/NDF");
				double chi2_electron = pdf_electron_rebin.at(i_hist)->Chi2Test(temp_hist_rebin.at(i_hist),"UUNORMCHI2/NDF");

				likelihood_electron_rebin.at(i_hist)->Fill(chi2_electron-chi2_muon);

				
				if (best_rebin_e.at(i_hist) == rebin[i_rebin]){
					lklh_values.at(i_hist) = chi2_electron - chi2_muon;
					if (i_hist == 0) {
						//b_charge->Fill();
						hist_PMTLikelihoodQ->Fill(lklh_values.at(i_hist));
						vec_lklh_charge_e.push_back(lklh_values.at(i_hist));
					}
					else if (i_hist == 1) {
						//b_time->Fill();
						hist_PMTLikelihoodT->Fill(lklh_values.at(i_hist));
						vec_lklh_time_e.push_back(lklh_values.at(i_hist));
					}
					else if (i_hist == 2) {
						//b_theta->Fill();
						hist_PMTLikelihoodTheta->Fill(lklh_values.at(i_hist));
						vec_lklh_theta_e.push_back(lklh_values.at(i_hist));
					}
					else if (i_hist == 3) {
						//b_phi->Fill();
						hist_PMTLikelihoodPhi->Fill(lklh_values.at(i_hist));
						vec_lklh_phi_e.push_back(lklh_values.at(i_hist));
					}
				}
				

			}
		}

		/*
		f_data_e->cd();
		classification_e->Write("",TObject::kOverwrite);

		if (best_rebin_e.at(0) == rebin[i_rebin]){
			hist_PMTLikelihoodQ->Write();
		}
		if (best_rebin_e.at(1) == rebin[i_rebin]){
			hist_PMTLikelihoodT->Write();
		}
		if (best_rebin_e.at(2) == rebin[i_rebin]){
			hist_PMTLikelihoodTheta->Write();
		}
		if (best_rebin_e.at(3) == rebin[i_rebin]){
			hist_PMTLikelihoodPhi->Write();
		}
		*/
		
		// Loop over muons-file

		for (int i_mu=0; i_mu < nentries_mu; i_mu++){
		
			temp_hist_charge_rebin->Reset();
			temp_hist_time_rebin->Reset();
			temp_hist_theta_rebin->Reset();
			temp_hist_phi_rebin->Reset();

			classification_mu->GetEntry(i_mu);

			for (int i_q = 0; i_q < (int) event_charge_mu->size(); i_q++){
				temp_hist_charge_rebin->Fill(event_charge_mu->at(i_q));
				temp_hist_time_rebin->Fill(event_time_mu->at(i_q));
				temp_hist_theta_rebin->Fill(event_theta_mu->at(i_q));
				temp_hist_phi_rebin->Fill(event_phi_mu->at(i_q));
			}

			num_pmts->Fill(event_charge_mu->size());

			for (int i_hist=0; i_hist< (int) pdf_hists_muon.size(); i_hist++){

				double chi2_muon = pdf_muon_rebin.at(i_hist)->Chi2Test(temp_hist_rebin.at(i_hist),"UUNORMCHI2/NDF");
				double chi2_electron = pdf_electron_rebin.at(i_hist)->Chi2Test(temp_hist_rebin.at(i_hist),"UUNORMCHI2/NDF");

				likelihood_muon_rebin.at(i_hist)->Fill(chi2_electron-chi2_muon);

				/*
				if (best_rebin_mu.at(i_hist) == rebin[i_rebin]){
					lklh_values_mu.at(i_hist) = chi2_electron - chi2_muon;
					if (i_hist == 0) {
						b_charge_mu->Fill();
						hist_PMTLikelihoodQ->Fill(lklh_values_mu.at(i_hist));
						vec_lklh_charge_mu.push_back(lklh_values_mu.at(i_hist));
					}
					else if (i_hist == 1) {
						b_time_mu->Fill();
						hist_PMTLikelihoodT->Fill(lklh_values_mu.at(i_hist));
						vec_lklh_time_mu.push_back(lklh_values_mu.at(i_hist));
					}
					else if (i_hist == 2) {
						b_theta_mu->Fill();
						hist_PMTLikelihoodTheta->Fill(lklh_values_mu.at(i_hist));
						vec_lklh_theta_mu.push_back(lklh_values_mu.at(i_hist));
					}
					else if (i_hist == 3) {
						b_phi_mu->Fill();
						hist_PMTLikelihoodPhi->Fill(lklh_values_mu.at(i_hist));
						vec_lklh_phi_mu.push_back(lklh_values_mu.at(i_hist));
					}
				}*/
				
				
			}
		}

		/*
		f_data_mu->cd();
		classification_mu->Write("",TObject::kOverwrite);

		if (best_rebin_mu.at(0) == rebin[i_rebin]){
			hist_PMTLikelihoodQ->Write();
		}
		if (best_rebin_mu.at(1) == rebin[i_rebin]){
			hist_PMTLikelihoodT->Write();
		}
		if (best_rebin_mu.at(2) == rebin[i_rebin]){
			hist_PMTLikelihoodTheta->Write();
		}
		if (best_rebin_mu.at(3) == rebin[i_rebin]){
			hist_PMTLikelihoodPhi->Write();
		}
		*/


		for (int i_hist=0; i_hist < (int) pdf_hists_muon.size(); i_hist++){

			likelihood_muon_rebin.at(i_hist)->GetXaxis()->SetTitle("#Delta #chi^{2} = #chi^{2}_{electron}-#chi^{2}_{muon}");
			likelihood_electron_rebin.at(i_hist)->GetXaxis()->SetTitle("#Delta #chi^{2} = #chi^{2}_{electron}-#chi^{2}_{muon}");
			likelihood_muon_rebin.at(i_hist)->SetLineColor(2);
			likelihood_muon_rebin.at(i_hist)->SetFillColor(kRed);
			likelihood_muon_rebin.at(i_hist)->SetFillStyle(3005);
			likelihood_electron_rebin.at(i_hist)->SetLineColor(4);
			likelihood_electron_rebin.at(i_hist)->SetFillColor(kBlue);
			likelihood_electron_rebin.at(i_hist)->SetFillStyle(3013);

			out->cd();
			likelihood_muon_rebin.at(i_hist)->Write();
			likelihood_electron_rebin.at(i_hist)->Write();
			likelihood_muon_rebin.at(i_hist)->Scale(1./likelihood_muon_rebin.at(i_hist)->Integral());
			likelihood_electron_rebin.at(i_hist)->Scale(1./likelihood_electron_rebin.at(i_hist)->Integral());
			likelihood_muon_rebin.at(i_hist)->SetStats(0);
			std::stringstream canvas_name;
			canvas_name << "canvas_"<<var_names[i_hist]<<"_rebin"<<rebin[i_rebin];
			TLegend *leg = new TLegend(0.75,0.75,0.9,0.9);
			TCanvas *c = new TCanvas(canvas_name.str().c_str(),canvas_name.str().c_str(),900,600);
			c->cd();
			std::stringstream hist_title;
			hist_title << "Likelihood "<<var_names[i_hist]<<" ("<<n_bins/rebin[i_rebin]<<" bins)";
			likelihood_muon_rebin.at(i_hist)->SetTitle(hist_title.str().c_str());
			likelihood_muon_rebin.at(i_hist)->Draw("HIST");
			likelihood_electron_rebin.at(i_hist)->Draw("same HIST");
			leg->AddEntry(likelihood_muon_rebin.at(i_hist),"muon","l");
			leg->AddEntry(likelihood_electron_rebin.at(i_hist),"electron","l");
			leg->Draw();
			c->Write();

			if (i_rebin == 0 && i_hist == 0) c->Print("PID_Likelihood.pdf(","pdf");
			else if (i_rebin == n_rebin-1 && i_hist == (int) pdf_hists_electron.size()-1) c->Print("PID_Likelihood.pdf)","pdf");
			else c->Print("PID_Likelihood.pdf","pdf");

			TH1F *likelihood_overlap = new TH1F(*likelihood_muon_rebin.at(i_hist));
			likelihood_overlap->SetNameTitle("likelihood_overlap","Likelihood overlap");
			likelihood_overlap->Reset("M");
			for (int i=1; i <= likelihood_overlap->GetNbinsX(); i++){
				likelihood_overlap->Fill(likelihood_overlap->GetBinCenter(i),TMath::Min(likelihood_muon_rebin.at(i_hist)->GetBinContent(i),likelihood_electron_rebin.at(i_hist)->GetBinContent(i)));
			}
			double overlap = likelihood_overlap->Integral();
			if (i_hist != (int) pdf_hists_muon.size()-1) file_overlap_emu << overlap <<",";
			else file_overlap_emu << overlap << std::endl;
			temp_overlap.push_back(overlap);

			pdf_muon_rebin.at(i_hist)->GetXaxis()->SetTitle(var_names[i_hist].c_str());
			pdf_electron_rebin.at(i_hist)->GetXaxis()->SetTitle(var_names[i_hist].c_str());
			pdf_muon_rebin.at(i_hist)->SetLineColor(2);
			pdf_electron_rebin.at(i_hist)->SetLineColor(4);
			std::stringstream pdf_canvas_name;
			pdf_canvas_name << "pdf_"<<var_names[i_hist]<<"_rebin"<<rebin[i_rebin];
			TLegend *leg_pdf = new TLegend(0.75,0.75,0.9,0.9);
			TCanvas *c_pdf = new TCanvas(pdf_canvas_name.str().c_str(),pdf_canvas_name.str().c_str(),900,600);
			c_pdf->cd();
			std::stringstream hist_pdf_title;
			hist_pdf_title << "PDF "<<var_names[i_hist] << " ("<<n_bins/rebin[i_rebin]<<" bins)";
			pdf_muon_rebin.at(i_hist)->SetTitle(hist_pdf_title.str().c_str());
			pdf_muon_rebin.at(i_hist)->SetStats(0);
			pdf_muon_rebin.at(i_hist)->Draw("HIST");
			pdf_electron_rebin.at(i_hist)->Draw("same HIST");
			leg_pdf->AddEntry(pdf_muon_rebin.at(i_hist),"muon","l");
			leg_pdf->AddEntry(pdf_electron_rebin.at(i_hist),"electron","l");
			leg_pdf->Draw();
			c_pdf->Write();

			if (i_rebin == 0 && i_hist == 0) c_pdf->Print("PID_Likelihood_pdf.pdf(","pdf");
			else if (i_rebin == n_rebin-1 && i_hist == (int) pdf_hists_electron.size()-1) c_pdf->Print("PID_Likelihood_pdf.pdf)","pdf");
			else c_pdf->Print("PID_Likelihood_pdf.pdf","pdf");

		}

		overlap_vector.push_back(temp_overlap);

	}

	for (int i_entry=0; i_entry < (int) vec_lklh_theta_e.size(); i_entry++){
		file_rings << vec_lklh_charge_e.at(i_entry) << "," << vec_lklh_time_e.at(i_entry) << "," << vec_lklh_theta_e.at(i_entry) << "," << vec_lklh_phi_e.at(i_entry) << std::endl;
	}

/*
	for (int i_entry=0; i_entry < (int) vec_lklh_theta_e.size(); i_entry++){
		file_electrons << vec_lklh_charge_e.at(i_entry) << "," << vec_lklh_time_e.at(i_entry) << "," << vec_lklh_theta_e.at(i_entry) << "," << vec_lklh_phi_e.at(i_entry) << std::endl;
	}

	*/
	/*
	for (int i_entry = 0; i_entry < (int) vec_lklh_theta_mu.size(); i_entry++){
		file_muons << vec_lklh_charge_mu.at(i_entry) << "," << vec_lklh_time_mu.at(i_entry) << "," << vec_lklh_theta_mu.at(i_entry) << "," << vec_lklh_phi_mu.at(i_entry) << std::endl;
	}*/

	out->cd();
	TGraph *gr_overlap_charge = new TGraph();
	TGraph *gr_overlap_time = new TGraph();
	TGraph *gr_overlap_theta = new TGraph();
	TGraph *gr_overlap_phi = new TGraph();

	for (int i_rebin=0; i_rebin < n_rebin; i_rebin++){
		gr_overlap_charge->SetPoint(i_rebin,n_bins/rebin[i_rebin],overlap_vector.at(i_rebin).at(0));
		gr_overlap_time->SetPoint(i_rebin,n_bins/rebin[i_rebin],overlap_vector.at(i_rebin).at(1));
		gr_overlap_theta->SetPoint(i_rebin,n_bins/rebin[i_rebin],overlap_vector.at(i_rebin).at(2));
		gr_overlap_phi->SetPoint(i_rebin,n_bins/rebin[i_rebin],overlap_vector.at(i_rebin).at(3));
	}

	gr_overlap_charge->SetTitle("PID Class Overlap charge");
	gr_overlap_charge->GetXaxis()->SetTitle("Number of bins");
	gr_overlap_charge->GetYaxis()->SetTitle("Class Overlap");
	gr_overlap_charge->SetLineColor(0);
	gr_overlap_charge->SetMarkerStyle(22);
	gr_overlap_charge->SetMarkerSize(0.5);
	gr_overlap_charge->Write("gr_overlap_charge");

	gr_overlap_time->SetTitle("PID Class Overlap time");
	gr_overlap_time->GetXaxis()->SetTitle("Number of bins");
	gr_overlap_time->GetYaxis()->SetTitle("Class Overlap");
	gr_overlap_time->SetLineColor(0);
	gr_overlap_time->SetMarkerStyle(22);
	gr_overlap_time->SetMarkerSize(0.5);
	gr_overlap_time->Write("gr_overlap_time");

	gr_overlap_theta->SetTitle("PID Class Overlap #theta");
	gr_overlap_theta->GetXaxis()->SetTitle("Number of bins");
	gr_overlap_theta->GetYaxis()->SetTitle("Class Overlap");
	gr_overlap_theta->SetLineColor(0);
	gr_overlap_theta->SetMarkerStyle(22);
	gr_overlap_theta->SetMarkerSize(0.5);
	gr_overlap_theta->Write("gr_overlap_theta");

	gr_overlap_phi->SetTitle("PID Class Overlap #phi");
	gr_overlap_phi->GetXaxis()->SetTitle("Number of bins");
	gr_overlap_phi->GetYaxis()->SetTitle("Class Overlap");
	gr_overlap_phi->SetLineColor(0);
	gr_overlap_phi->SetMarkerStyle(22);
	gr_overlap_phi->SetMarkerSize(0.5);
	gr_overlap_phi->Write("gr_overlap_phi");

	file_overlap_emu.close();
	//file_electrons.close();
	//file_muons.close();
	out->Close();
	f_data_rings->Close();
	f_data_mu->Close();


}

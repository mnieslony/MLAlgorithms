void create_pdfs_emu(){


	int n_bins = 500;

	TH1F *pdf_mu_charge = new TH1F("pdf_beamlike_muon_charge","pdf_beamlike_muon_charge",n_bins,10,100);
	TH1F *pdf_mu_time = new TH1F("pdf_beamlike_muon_time","pdf_beamlike_muon_time",n_bins,0,20);
	TH1F *pdf_mu_thetaB = new TH1F("pdf_beamlike_muon_thetaB","pdf_beamlike_muon_thetaB",n_bins,-1.2,2.2);
	TH1F *pdf_mu_phiB = new TH1F("pdf_beamlike_muon_phiB","pdf_beamlike_muon_phiB",n_bins,-3.2,3.2);
	TH1F *pdf_e_charge = new TH1F("pdf_beamlike_electron_charge","pdf_beamlike_electron_charge",n_bins,10,100);
	TH1F *pdf_e_time = new TH1F("pdf_beamlike_electron_time","pdf_beamlike_electron_time",n_bins,0,20);
	TH1F *pdf_e_thetaB = new TH1F("pdf_beamlike_electron_thetaB","pdf_beamlike_electron_thetaB",n_bins,-1.2,2.2);
	TH1F *pdf_e_phiB = new TH1F("pdf_beamlike_electron_phiB","pdf_beamlike_electron_phiB",n_bins,-3.2,3.2);

	// Electrons

	TFile *file_electrons = new TFile("data/beamlike_electrons_DigitThr10_0_276.root","READ");
	TTree *classification_e = (TTree*) file_electrons->Get("classification_tree");
	int nentries = classification_e->GetEntries();
	std::vector<double>* event_charge = new std::vector<double>;
	std::vector<double>* event_time = new std::vector<double>;
	std::vector<double>* event_theta = new std::vector<double>;
	std::vector<double>* event_phi = new std::vector<double>;
	classification_e->SetBranchAddress("PMTQVector",&event_charge);
	classification_e->SetBranchAddress("PMTTVector",&event_time);
	classification_e->SetBranchAddress("PMTThetaBaryVector",&event_theta);
	classification_e->SetBranchAddress("PMTPhiBaryVector",&event_phi);

	for (int i_e=0; i_e < nentries; i_e++){
		classification_e->GetEntry(i_e);
		for (int i_q = 0 ; i_q < event_charge->size(); i_q++){
			pdf_e_charge->Fill(event_charge->at(i_q));
			pdf_e_time->Fill(event_time->at(i_q));
			pdf_e_thetaB->Fill(event_theta->at(i_q));
			pdf_e_phiB->Fill(event_phi->at(i_q));
		}
	}	
	
	pdf_e_charge->Sumw2();
	pdf_e_time->Sumw2();
	pdf_e_thetaB->Sumw2();
	pdf_e_phiB->Sumw2();

	pdf_e_charge->Scale(1./pdf_e_charge->Integral());
	pdf_e_time->Scale(1./pdf_e_time->Integral());
	pdf_e_thetaB->Scale(1./pdf_e_thetaB->Integral());
	pdf_e_phiB->Scale(1./pdf_e_phiB->Integral());

	// Muons

	TFile *file_muons = new TFile("data/beamlike_muons_DigitThr10_0_498.root","READ");
	TTree *classification_mu = (TTree*) file_muons->Get("classification_tree");
	int nentries_mu = classification_mu->GetEntries();
	std::vector<double>* event_charge_mu = new std::vector<double>;
	std::vector<double>* event_time_mu = new std::vector<double>;
	std::vector<double>* event_theta_mu = new std::vector<double>;
	std::vector<double>* event_phi_mu = new std::vector<double>;
	classification_mu->SetBranchAddress("PMTQVector",&event_charge_mu);
	classification_mu->SetBranchAddress("PMTTVector",&event_time_mu);
	classification_mu->SetBranchAddress("PMTThetaBaryVector",&event_theta_mu);
	classification_mu->SetBranchAddress("PMTPhiBaryVector",&event_phi_mu);

	for (int i_mu=0; i_mu < nentries_mu; i_mu++){
		classification_mu->GetEntry(i_mu);
		for (int i_q = 0 ; i_q < event_charge_mu->size(); i_q++){
			pdf_mu_charge->Fill(event_charge_mu->at(i_q));
			pdf_mu_time->Fill(event_time_mu->at(i_q));
			pdf_mu_thetaB->Fill(event_theta_mu->at(i_q));
			pdf_mu_phiB->Fill(event_phi_mu->at(i_q));
		}
	}

	pdf_mu_charge->Sumw2();
	pdf_mu_time->Sumw2();
	pdf_mu_thetaB->Sumw2();
	pdf_mu_phiB->Sumw2();

	pdf_mu_charge->Scale(1./pdf_mu_charge->Integral());
	pdf_mu_time->Scale(1./pdf_mu_time->Integral());
	pdf_mu_thetaB->Scale(1./pdf_mu_thetaB->Integral());
	pdf_mu_phiB->Scale(1./pdf_mu_phiB->Integral());

	// Write pdfs to file

	std::stringstream outfile_name;
	outfile_name << "pdf_beamlike_emu_"<<n_bins<<"bins.root";

	TFile *out = new TFile(outfile_name.str().c_str(),"RECREATE");
	out->cd();
	pdf_e_charge->Write();
	pdf_e_time->Write();
	pdf_e_thetaB->Write();
	pdf_e_phiB->Write();
	pdf_mu_charge->Write();
	pdf_mu_time->Write();
	pdf_mu_thetaB->Write();
	pdf_mu_phiB->Write();
	out->Close();

}

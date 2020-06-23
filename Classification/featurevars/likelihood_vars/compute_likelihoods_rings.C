void compute_likelihoods_rings(){


	TFile *f_pdf = new TFile("pdfs/pdf_beam_rings_500bins.root","READ");
	TH1F *pdf_single_charge = (TH1F*) f_pdf->Get("pdf_beam_single_charge");
	TH1F *pdf_multi_charge = (TH1F*) f_pdf->Get("pdf_beam_multi_charge");
	TH1F *pdf_single_time = (TH1F*) f_pdf->Get("pdf_beam_single_time");
	TH1F *pdf_multi_time = (TH1F*) f_pdf->Get("pdf_beam_multi_time");
	TH1F *pdf_single_theta = (TH1F*) f_pdf->Get("pdf_beam_single_thetaB");
	TH1F *pdf_multi_theta = (TH1F*) f_pdf->Get("pdf_beam_multi_thetaB");
	TH1F *pdf_single_phi = (TH1F*) f_pdf->Get("pdf_beam_single_phiB");
	TH1F *pdf_multi_phi = (TH1F*) f_pdf->Get("pdf_beam_multi_phiB");
	
	//Vector of pdfs
	std::vector<TH1F*> pdf_hists_single = {pdf_single_charge, pdf_single_time, pdf_single_theta, pdf_single_phi};
	std::vector<TH1F*> pdf_hists_multi = {pdf_multi_charge, pdf_multi_time, pdf_multi_theta, pdf_multi_phi};

	//Output ROOT file
	TFile *out = new TFile("likelihood_result_rings_500bins.root","RECREATE");
	int n_bins = 500;

	// Test histograms (drawn event distributions, likelihood value results)
	TH1F *test_single_charge = new TH1F("test_single_charge","Test distribution Q single-ring",n_bins,10,100);
	TH1F *test_multi_charge = new TH1F("test_multi_charge","Test distribution Q multi-ring",n_bins,10,100);
	TH1F *test_single_time = new TH1F("test_single_time","Test distribution T single-ring",n_bins,0,20);
	TH1F *test_multi_time = new TH1F("test_multi_time","Test distribution T multi-ring",n_bins,0,20);
	TH1F *test_single_theta = new TH1F("test_single_theta","Test distribution Theta single-ring",n_bins,-1.2,2.2);
	TH1F *test_multi_theta = new TH1F("test_multi_theta","Test distribution Theta multi-ring",n_bins,-1.2,2.2);
	TH1F *test_single_phi = new TH1F("test_single_phi","Test distribution Phi single-ring",n_bins,-3.2,3.2);
	TH1F *test_multi_phi = new TH1F("test_multi_phi","Test distribution Phi multi-ring",n_bins,-3.2,3.2);
	TH1F *likelihood_test_single_charge = new TH1F("likelihood_test_single_charge","Likelihood_test_single_charge",200,-5,5);
	TH1F *likelihood_test_multi_charge = new TH1F ("likelihood_test_multi_charge","Likelihood_test_multi_charge",200,-5,5);
	TH1F *likelihood_test_single_time = new TH1F("likelihood_test_single_time","Likelihood_test_single_time",200,-5,5);
	TH1F *likelihood_test_multi_time = new TH1F ("likelihood_test_multi_time","Likelihood_test_multi_time",200,-5,5);
	TH1F *likelihood_test_single_theta = new TH1F("likelihood_test_single_charge","Likelihood_test_single_theta",200,-5,5);
	TH1F *likelihood_test_multi_theta = new TH1F ("likelihood_test_multi_charge","Likelihood_test_multi_theta",200,-5,5);
	TH1F *likelihood_test_single_phi = new TH1F("likelihood_test_single_phi","Likelihood_test_single_phi",200,-5,5);
	TH1F *likelihood_test_multi_phi = new TH1F ("likelihood_test_multi_phi","Likelihood_test_multi_phi",200,-5,5);
	
	//Vector of likelihood & event histograms
	std::vector<TH1F*> test_hists_single = {test_single_charge, test_single_time, test_single_theta, test_single_phi};
	std::vector<TH1F*> test_hists_multi = {test_multi_charge, test_multi_time, test_multi_theta, test_multi_phi};
	std::vector<TH1F*> likelihood_test_hists_single = {likelihood_test_single_charge, likelihood_test_single_time, likelihood_test_single_theta, likelihood_test_single_phi};
	std::vector<TH1F*> likelihood_test_hists_multi = {likelihood_test_multi_charge, likelihood_test_multi_time, likelihood_test_multi_theta, likelihood_test_multi_phi};

	// Test theoretical performance of likelihood values by drawing events from the pdfs
	const int n_rebin = 8;
	Int_t rebin[n_rebin]={1,2,5,10,20,25,50,100};
	std::vector<std::string> var_names = {"charge","time","theta","phi"};
	int n_toy_events = 10000;
	int nhits_per_event = 20;

	/*
	for (int i_rebin = 0; i_rebin < n_rebin; i_rebin++) {

		for (int i_hist=0; i_hist < (int) pdf_hists_single.size(); i_hist++){

			TH1F *lklh_single = (TH1F*) likelihood_test_hists_single.at(i_hist)->Clone();
			TH1F *lklh_multi = (TH1F*) likelihood_test_hists_multi.at(i_hist)->Clone();
			lklh_single->Reset();
			lklh_multi->Reset();
			std::stringstream histname_single, histname_multi;
			histname_single << "likelihood_test_single_"<<var_names[i_hist]<<"_rebin"<<rebin[i_rebin];
			histname_multi << "likelihood_test_multi_"<<var_names[i_hist]<<"_rebin"<<rebin[i_rebin];
			lklh_single->SetName(histname_single.str().c_str());
			lklh_multi->SetName(histname_multi.str().c_str());

			TH1F *temp_pdf_single_rebin = (TH1F*) ((TH1F*) pdf_hists_single.at(i_hist)->Clone())->Rebin(rebin[i_rebin]);
			TH1F *temp_pdf_multi_rebin = (TH1F*) ((TH1F*) pdf_hists_multi.at(i_hist)->Clone())->Rebin(rebin[i_rebin]);
			TH1F *temp_test_single_rebin = (TH1F*) ((TH1F*) test_hists_single.at(i_hist)->Clone())->Rebin(rebin[i_rebin]);
			TH1F *temp_test_multi_rebin = (TH1F*) ((TH1F*) test_hists_multi.at(i_hist)->Clone())->Rebin(rebin[i_rebin]);
			
			for  (int i_toy=0; i_toy < n_toy_events; i_toy++){
				temp_test_single_rebin->Reset();
				temp_test_multi_rebin->Reset();
				for (int i_hit=0; i_hit < nhits_per_event; i_hit++){
					temp_test_single_rebin->Fill(temp_pdf_single_rebin->GetRandom());
					temp_test_multi_rebin->Fill(temp_pdf_multi_rebin->GetRandom());
				}

				double chi2_Single_single = temp_pdf_single_rebin->Chi2Test(temp_test_single_rebin,"UUNORMCHI2/NDF");
				double chi2_Single_multi = temp_pdf_multi_rebin->Chi2Test(temp_test_single_rebin,"UUNORMCHI2/NDF");
				double chi2_Multi_single = temp_pdf_single_rebin->Chi2Test(temp_test_multi_rebin,"UUNORMCHI2/NDF");
				double chi2_Multi_multi = temp_pdf_multi_rebin->Chi2Test(temp_test_multi_rebin,"UUNORMCHI2/NDF");

				lklh_single->Fill(chi2_Single_multi-chi2_Single_single);
				lklh_multi->Fill(chi2_Multi_multi-chi2_Multi_single);

			}

			lklh_single->GetXaxis()->SetTitle("#Delta #chi^{2} = #chi^{2}_{multi}-#chi^{2}_{single}");
			lklh_multi->GetXaxis()->SetTitle("#Delta #chi^{2} = #chi^{2}_{multi}-#chi^{2}_{single}");
			lklh_single->SetLineColor(2);
			lklh_single->SetFillColor(kRed);
			lklh_single->SetFillStyle(3005);
			lklh_multi->SetLineColor(4);
			lklh_multi->SetFillColor(kBlue);
			lklh_multi->SetFillStyle(3013);

			out->cd();
			lklh_single->Write();
			lklh_multi->Write();

			std::stringstream canvas_name_test;
			canvas_name_test << "canvas_test_"<<var_names[i_hist]<<"rebin"<<rebin[i_rebin];

			TLegend *leg_test = new TLegend(0.75,0.75,0.9,0.9);
			TCanvas *c_test = new TCanvas(canvas_name_test.str().c_str(),canvas_name_test.str().c_str(),900,600);
			c_test->cd();
			lklh_single->SetStats(0);
			std::stringstream hist_title;
			hist_title <<"Ring Counting Likelihood "<<var_names[i_hist]<<" ("<<n_bins/rebin[i_rebin]<<" bins)";
			lklh_single->SetTitle(hist_title.str().c_str());
			lklh_single->Draw("HIST");
			lklh_multi->Draw("same HIST");
			leg_test->AddEntry(lklh_single,"single-ring","l");
			leg_test->AddEntry(lklh_multi,"multi-ring","l");
			leg_test->Draw();
			c_test->Write();

			if (i_rebin == 0 && i_hist == 0) c_test->Print("RingCounting_Likelihood_Test.pdf(","pdf");
			else if (i_rebin == n_rebin-1 && i_hist == (int) pdf_hists_single.size()-1) c_test->Print("RingCounting_Likelihood_Test.pdf)","pdf");
			else c_test->Print("RingCounting_Likelihood_Test.pdf","pdf");

		}

	}
*/

	//TFile *f_data_rings = new TFile("data/beam_DigitThr10_0_4996.root","READ");
	TFile *f_data_rings = new TFile("../../data_new.nosync/beamlike_electrons_DigitThr10_0_276.root","READ");

	TTree *classification_rings = (TTree*) f_data_rings->Get("classification_tree");

	int nentries = classification_rings->GetEntries();
	std::vector<double>* event_charge = new std::vector<double>;
	std::vector<double>* event_time = new std::vector<double>;
	std::vector<double>* event_theta = new std::vector<double>;
	std::vector<double>* event_phi = new std::vector<double>;
	bool fMCMultiRing;

	classification_rings->SetBranchAddress("PMTQVector",&event_charge);
	classification_rings->SetBranchAddress("PMTTVector",&event_time);
	classification_rings->SetBranchAddress("PMTThetaBaryVector",&event_theta);
	classification_rings->SetBranchAddress("PMTPhiBaryVector",&event_phi);	
	classification_rings->SetBranchAddress("MCMultiRing",&fMCMultiRing);

	//Add newly calculated classification variables
	double lklh_charge, lklh_time, lklh_theta, lklh_phi;
	std::vector<double> lklh_values = {lklh_charge, lklh_time, lklh_theta, lklh_phi};
	/*
	TBranch *b_charge = classification_rings->Branch("PMTLikelihoodQRings",&lklh_values.at(0));
	TBranch *b_time = classification_rings->Branch("PMTLikelihoodTRings",&lklh_values.at(1));
	TBranch *b_theta = classification_rings->Branch("PMTLikelihoodThetaRings",&lklh_values.at(2));
	TBranch *b_phi = classification_rings->Branch("PMTLikelihoodPhiRings",&lklh_values.at(3));*/

	TH1F *hist_PMTLikelihoodQRings = new TH1F("hist_PMTLikelihoodQRings","hist_PMTLikelihoodQRings",200,-5,5);
	TH1F *hist_PMTLikelihoodTRings = new TH1F("hist_PMTLikelihoodTRings","hist_PMTLikelihoodTRings",200,-5,5);
	TH1F *hist_PMTLikelihoodThetaRings = new TH1F("hist_PMTLikelihoodThetaRings","hist_PMTLikelihoodThetaRings",200,-5,5);
	TH1F *hist_PMTLikelihoodPhiRings = new TH1F("hist_PMTLikelihoodPhiRings","hist_PMTLikelihoodPhiRings",200,-5,5);

	TH1F *likelihood_single_charge = new TH1F("likelihood_single_charge","Likelihood Q single-ring",200,-5,5);
	TH1F *likelihood_multi_charge = new TH1F("likelihood_multi_charge","Likelihood Q multi-ring",200,-5,5);
	TH1F *likelihood_single_time = new TH1F("likelihood_single_time","Likelihood T single-ring",200,-5,5);
	TH1F *likelihood_multi_time = new TH1F("likelihood_multi_time","Likelihood T multi-ring",200,-5,5);
	TH1F *likelihood_single_theta = new TH1F("likelihood_single_theta","Likelihood Theta single-ring",200,-5,5);
	TH1F *likelihood_multi_theta = new TH1F("likelihood_multi_theta","Likelihood Theta multi-ring",200,-5,5);
	TH1F *likelihood_single_phi = new TH1F("likelihood_single_phi","Likelihood Phi single-ring",200,-5,5);
	TH1F *likelihood_multi_phi = new TH1F("likelihood_multi_phi","Likelihood Phi multi-ring",200,-5,5);

	std::vector<TH1F*> likelihood_single = {likelihood_single_charge, likelihood_single_time, likelihood_single_theta, likelihood_single_phi};
	std::vector<TH1F*> likelihood_multi = {likelihood_multi_charge, likelihood_multi_time, likelihood_multi_theta, likelihood_multi_phi};

	TH1F *temp_hist_charge = new TH1F("temp_hist_charge","Event hist charge",n_bins,10,100);
	TH1F *temp_hist_time = new TH1F("temp_hist_time","Event hist time",n_bins,0,20);
	TH1F *temp_hist_theta = new TH1F("temp_hist_theta","Event hist theta",n_bins,-1.2,2.2);
	TH1F *temp_hist_phi = new TH1F("temp_hist_phi","Event hist phi",n_bins,-3.2,3.2);
	TH1F *num_pmts = new TH1F("num_pmts","Number of PMTs",130,0,130);

	std::vector<TH1F*> temp_hists = {temp_hist_charge, temp_hist_time, temp_hist_theta, temp_hist_phi};

	ofstream file_rings("beamlikev1_electrons_rings_likelihood.csv");
	file_rings << "LikelihoodQRing,LikelihoodTRing,LikelihoodThetaRing,LikelihoodPhiRing"<<std::endl;
	ofstream file_overlap("likelihood_overlap_bin.csv");
	file_overlap << "Bins,OverlapQ,OverlapT,OverlapTheta,OverlapPhi"<<std::endl;
	std::vector<std::vector<double>> overlap_vector;
	std::vector<int> best_rebin = {50,1,1,1};
	std::vector<double> vec_lklh_theta, vec_lklh_phi, vec_lklh_charge, vec_lklh_time;


	for (int i_rebin = 0; i_rebin < n_rebin; i_rebin++) {


		file_overlap << rebin[i_rebin] << ",";

		std::vector<double> temp_overlap;

		TH1F *temp_hist_charge_rebin = (TH1F*) ((TH1F*)temp_hist_charge->Clone())->Rebin(rebin[i_rebin]);
		TH1F *temp_hist_time_rebin = (TH1F*) ((TH1F*)temp_hist_time->Clone())->Rebin(rebin[i_rebin]);
		TH1F *temp_hist_theta_rebin = (TH1F*) ((TH1F*)temp_hist_theta->Clone())->Rebin(rebin[i_rebin]);
		TH1F *temp_hist_phi_rebin = (TH1F*) ((TH1F*)temp_hist_phi->Clone())->Rebin(rebin[i_rebin]);

		std::vector<TH1F*> temp_hist_rebin={temp_hist_charge_rebin,temp_hist_time_rebin,temp_hist_theta_rebin,temp_hist_phi_rebin};
		std::vector<TH1F*> likelihood_single_rebin;
		std::vector<TH1F*> likelihood_multi_rebin;
		std::vector<TH1F*> pdf_single_rebin;
		std::vector<TH1F*> pdf_multi_rebin;

		for (int i_hist=0; i_hist < (int) pdf_hists_single.size(); i_hist++){
			TH1F *lklhd_single = (TH1F*) likelihood_single.at(i_hist)->Clone();
			TH1F *lklhd_multi = (TH1F*) likelihood_multi.at(i_hist)->Clone();
			lklhd_single->Reset();
			lklhd_multi->Reset();
			std::stringstream histname_single, histname_multi;
			histname_single << "likelihood_single_"<<var_names[i_hist]<<"_rebin"<<rebin[i_rebin];
			histname_multi << "likelihood_multi_"<<var_names[i_hist]<<"_rebin"<<rebin[i_rebin];
			lklhd_single->SetName(histname_single.str().c_str());
			lklhd_multi->SetName(histname_multi.str().c_str());
			likelihood_single_rebin.push_back(lklhd_single);
			likelihood_multi_rebin.push_back(lklhd_multi);

			TH1F *temp_pdf_single_rebin = (TH1F*) ((TH1F*) pdf_hists_single.at(i_hist)->Clone())->Rebin(rebin[i_rebin]);
			TH1F *temp_pdf_multi_rebin = (TH1F*) ((TH1F*) pdf_hists_multi.at(i_hist)->Clone())->Rebin(rebin[i_rebin]);
			
			pdf_single_rebin.push_back(temp_pdf_single_rebin);
			pdf_multi_rebin.push_back(temp_pdf_multi_rebin);
		}

		for (int i=0; i < nentries; i++){

			temp_hist_charge_rebin->Reset();
			temp_hist_time_rebin->Reset();
			temp_hist_theta_rebin->Reset();
			temp_hist_phi_rebin->Reset();

			classification_rings->GetEntry(i);

			for (int i_q = 0; i_q < (int) event_charge->size(); i_q++){
				temp_hist_charge_rebin->Fill(event_charge->at(i_q));
				temp_hist_time_rebin->Fill(event_time->at(i_q));
				temp_hist_theta_rebin->Fill(event_theta->at(i_q));
				temp_hist_phi_rebin->Fill(event_phi->at(i_q));
			}

			for (int i_hist=0; i_hist< (int) pdf_hists_single.size(); i_hist++){

				if (best_rebin.at(i_hist) != rebin[i_rebin]) continue;


				double chi2_single = pdf_single_rebin.at(i_hist)->Chi2Test(temp_hist_rebin.at(i_hist),"UUNORMCHI2/NDF");
				double chi2_multi = pdf_multi_rebin.at(i_hist)->Chi2Test(temp_hist_rebin.at(i_hist),"UUNORMCHI2/NDF");

				if (fMCMultiRing) likelihood_multi_rebin.at(i_hist)->Fill(chi2_multi-chi2_single);
				else likelihood_single_rebin.at(i_hist)->Fill(chi2_multi-chi2_single);

				
				if (best_rebin.at(i_hist) == rebin[i_rebin]){
					lklh_values.at(i_hist) = chi2_multi - chi2_single;
					if (i_hist == 0) {
						//b_charge->Fill();
						hist_PMTLikelihoodQRings->Fill(lklh_values.at(i_hist));
						vec_lklh_charge.push_back(lklh_values.at(i_hist));
					}
					else if (i_hist == 1) {
						//b_time->Fill();
						hist_PMTLikelihoodTRings->Fill(lklh_values.at(i_hist));
						vec_lklh_time.push_back(lklh_values.at(i_hist));
					}
					else if (i_hist == 2) {
						//b_theta->Fill();
						hist_PMTLikelihoodThetaRings->Fill(lklh_values.at(i_hist));
						vec_lklh_theta.push_back(lklh_values.at(i_hist));
					}
					else if (i_hist == 3) {
						//b_phi->Fill();
						hist_PMTLikelihoodPhiRings->Fill(lklh_values.at(i_hist));
						vec_lklh_phi.push_back(lklh_values.at(i_hist));
					}
				}
				
			}
		}

/*
		f_data_rings->cd();
		classification_rings->Write("",TObject::kOverwrite);

		if (best_rebin.at(0) == rebin[i_rebin]){
			hist_PMTLikelihoodQRings->Write();
		}
		if (best_rebin.at(1) == rebin[i_rebin]){
			hist_PMTLikelihoodTRings->Write();
		}
		if (best_rebin.at(2) == rebin[i_rebin]){
			hist_PMTLikelihoodThetaRings->Write();
		}
		if (best_rebin.at(3) == rebin[i_rebin]){
			hist_PMTLikelihoodPhiRings->Write();
		}
*/
		for (int i_hist=0; i_hist < (int) pdf_hists_single.size(); i_hist++){

			if (best_rebin.at(i_hist) != rebin[i_rebin]) continue;


			likelihood_single_rebin.at(i_hist)->GetXaxis()->SetTitle("#Delta #chi^{2} = #chi^{2}_{multi}-#chi^{2}_{single}");
			likelihood_multi_rebin.at(i_hist)->GetXaxis()->SetTitle("#Delta #chi^{2} = #chi^{2}_{multi}-#chi^{2}_{single}");
			likelihood_single_rebin.at(i_hist)->SetLineColor(2);
			likelihood_single_rebin.at(i_hist)->SetFillColor(kRed);
			likelihood_single_rebin.at(i_hist)->SetFillStyle(3005);
			likelihood_multi_rebin.at(i_hist)->SetLineColor(4);
			likelihood_multi_rebin.at(i_hist)->SetFillColor(kBlue);
			likelihood_multi_rebin.at(i_hist)->SetFillStyle(3013);

			out->cd();
			likelihood_single_rebin.at(i_hist)->Write();
			likelihood_multi_rebin.at(i_hist)->Write();
			likelihood_single_rebin.at(i_hist)->Scale(1./likelihood_single_rebin.at(i_hist)->Integral());
			likelihood_multi_rebin.at(i_hist)->Scale(1./likelihood_multi_rebin.at(i_hist)->Integral());
			likelihood_single_rebin.at(i_hist)->SetStats(0);
			std::stringstream canvas_name;
			canvas_name << "canvas_"<<var_names[i_hist]<<"_rebin"<<rebin[i_rebin];
			TLegend *leg = new TLegend(0.75,0.75,0.9,0.9);
			TCanvas *c = new TCanvas(canvas_name.str().c_str(),canvas_name.str().c_str(),900,600);
			c->cd();
			std::stringstream hist_title;
			hist_title << "Likelihood "<<var_names[i_hist] <<" ("<<n_bins/rebin[i_rebin]<<" bins)";
			likelihood_single_rebin.at(i_hist)->SetTitle(hist_title.str().c_str());
			likelihood_single_rebin.at(i_hist)->Draw("HIST");
			likelihood_multi_rebin.at(i_hist)->Draw("same HIST");
			leg->AddEntry(likelihood_single_rebin.at(i_hist),"single-ring","l");
			leg->AddEntry(likelihood_multi_rebin.at(i_hist),"multi-ring","l");
			leg->Draw();
			c->Write();

			if (i_rebin == 0 && i_hist == 0) c->Print("Beamlikev2_Muon_RingCounting_Likelihood.pdf(","pdf");
			else if (i_rebin == n_rebin-1 && i_hist == (int) pdf_hists_single.size()-1) c->Print("Beamlikev2_Muon_RingCounting_Likelihood.pdf)","pdf");
			else c->Print("Beamlikev2_Muon_RingCounting_Likelihood.pdf","pdf");

			TH1F *likelihood_overlap = new TH1F(*likelihood_single_rebin.at(i_hist));
			likelihood_overlap->SetNameTitle("likelihood_overlap","Likelihood overlap");
			likelihood_overlap->Reset("M");
			for (int i=1; i <= likelihood_overlap->GetNbinsX(); i++){
				likelihood_overlap->Fill(likelihood_overlap->GetBinCenter(i),TMath::Min(likelihood_single_rebin.at(i_hist)->GetBinContent(i),likelihood_multi_rebin.at(i_hist)->GetBinContent(i)));
			}
			double overlap = likelihood_overlap->Integral();
			if (i_hist != (int) pdf_hists_single.size()-1) file_overlap << overlap <<",";
			else file_overlap << overlap << std::endl;
			temp_overlap.push_back(overlap);

			pdf_single_rebin.at(i_hist)->GetXaxis()->SetTitle(var_names[i_hist].c_str());
			pdf_multi_rebin.at(i_hist)->GetXaxis()->SetTitle(var_names[i_hist].c_str());
			pdf_single_rebin.at(i_hist)->SetLineColor(2);
			pdf_multi_rebin.at(i_hist)->SetLineColor(4);
			std::stringstream pdf_canvas_name;
			pdf_canvas_name << "pdf_"<<var_names[i_hist]<<"_rebin"<<rebin[i_rebin];
			TLegend *leg_pdf = new TLegend(0.75,0.75,0.9,0.9);
			TCanvas *c_pdf = new TCanvas(pdf_canvas_name.str().c_str(),pdf_canvas_name.str().c_str(),900,600);
			c_pdf->cd();
			std::stringstream hist_pdf_title;
			hist_pdf_title << "PDF "<<var_names[i_hist] <<" ("<<n_bins/rebin[i_rebin]<<" bins)";
			pdf_single_rebin.at(i_hist)->SetTitle(hist_pdf_title.str().c_str());
			pdf_single_rebin.at(i_hist)->SetStats(0);
			pdf_single_rebin.at(i_hist)->Draw("HIST");
			pdf_multi_rebin.at(i_hist)->Draw("same HIST");
			leg_pdf->AddEntry(pdf_single_rebin.at(i_hist),"single-ring","l");
			leg_pdf->AddEntry(pdf_multi_rebin.at(i_hist),"multi-ring","l");
			leg_pdf->Draw();
			c_pdf->Write();

			if (i_rebin == 0 && i_hist == 0) c_pdf->Print("RingCounting_Likelihood_pdf.pdf(","pdf");
			else if (i_rebin == n_rebin-1 && i_hist == (int) pdf_hists_single.size()-1) c_pdf->Print("RingCounting_Likelihood_pdf.pdf)","pdf");
			else c_pdf->Print("RingCounting_Likelihood_pdf.pdf","pdf");

		}

		overlap_vector.push_back(temp_overlap);

	}


	for (int i_entry=0; i_entry < (int) vec_lklh_theta.size(); i_entry++){
		file_rings << vec_lklh_charge.at(i_entry) << "," << vec_lklh_time.at(i_entry) << "," << vec_lklh_theta.at(i_entry) << "," << vec_lklh_phi.at(i_entry) << std::endl;
	}



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

	gr_overlap_charge->SetTitle("Class Overlap charge");
	gr_overlap_charge->GetXaxis()->SetTitle("Number of bins");
	gr_overlap_charge->GetYaxis()->SetTitle("Class Overlap");
	gr_overlap_charge->SetLineColor(0);
	gr_overlap_charge->SetMarkerStyle(22);
	gr_overlap_charge->SetMarkerSize(0.5);
	gr_overlap_charge->Write("gr_overlap_charge");

	gr_overlap_time->SetTitle("Class Overlap time");
	gr_overlap_time->GetXaxis()->SetTitle("Number of bins");
	gr_overlap_time->GetYaxis()->SetTitle("Class Overlap");
	gr_overlap_time->SetLineColor(0);
	gr_overlap_time->SetMarkerStyle(22);
	gr_overlap_time->SetMarkerSize(0.5);
	gr_overlap_time->Write("gr_overlap_time");

	gr_overlap_theta->SetTitle("Class Overlap #theta");
	gr_overlap_theta->GetXaxis()->SetTitle("Number of bins");
	gr_overlap_theta->GetYaxis()->SetTitle("Class Overlap");
	gr_overlap_theta->SetLineColor(0);
	gr_overlap_theta->SetMarkerStyle(22);
	gr_overlap_theta->SetMarkerSize(0.5);
	gr_overlap_theta->Write("gr_overlap_theta");

	gr_overlap_phi->SetTitle("Class Overlap #phi");
	gr_overlap_phi->GetXaxis()->SetTitle("Number of bins");
	gr_overlap_phi->GetYaxis()->SetTitle("Class Overlap");
	gr_overlap_phi->SetLineColor(0);
	gr_overlap_phi->SetMarkerStyle(22);
	gr_overlap_phi->SetMarkerSize(0.5);
	gr_overlap_phi->Write("gr_overlap_phi");


	file_overlap.close();
	//file_rings.close();
	out->Close();
	f_data_rings->Close();
	f_pdf->Close();

}

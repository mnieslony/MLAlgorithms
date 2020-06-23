void compare_emu(const char* efile, const char* mufile, const char* outfile, const char* configfile, const char* label1, const char* label2){


	std::cout <<"compare_emu_pid: Reading in files "<<efile<<" and "<<mufile<<std::endl;
	bool execute_program = true;

	ifstream file1(efile);
	if (!file1.good()) {
		std::cout <<"Specified electron file "<<efile<<" does not exist! Quit program..."<<std::endl;
		execute_program = false;
	}
	ifstream file2(mufile);
	if (!file2.good()){
		std::cout <<"Specified muon file "<<mufile<<" does not exist! Quit program..."<<std::endl;
		execute_program = false;
	}	

	std::cout <<"Reading in variable configuration file "<<configfile<<std::endl;
	ifstream file_config(configfile);
	if (!file_config.good()) {
		std::cout <<"Specified variable configuration file "<<configfile<<" does not exist! Quit program..."<<std::endl;
		execute_program = false;
	}


	if (execute_program){

		TFile *f_e = new TFile(efile,"READ");
		TFile *f_mu = new TFile(mufile,"READ");
		TFile *f_out = new TFile(outfile,"RECREATE");

		std::vector<std::string> vector_varnames;
		std::string temp_string;
		while (!file_config.eof()){
			file_config >> temp_string;
			if (file_config.eof()) break;
			vector_varnames.push_back(temp_string);
		}
		file_config.close();

		TH1F *hist_temp_e = nullptr;
		TH1F *hist_temp_mu = nullptr;
		
		std::vector<TCanvas*> vector_canvas;
        TLegend *leg = new TLegend(0.75,0.75,0.9,0.9);

		for (unsigned int i_var = 0; i_var < vector_varnames.size(); i_var++){

			std::cout <<vector_varnames.at(i_var)<<std::endl;
			std::stringstream ss_histname;
			ss_histname << "hist_"<<vector_varnames.at(i_var);
			hist_temp_e = (TH1F*) f_e->Get(ss_histname.str().c_str());
            hist_temp_mu = (TH1F*) f_mu->Get(ss_histname.str().c_str());
            hist_temp_e->Scale(1./hist_temp_e->Integral());
            hist_temp_mu->Scale(1./hist_temp_mu->Integral());
            hist_temp_mu->SetLineColor(2);
            hist_temp_e->SetStats(0);

			double max_e = hist_temp_e->GetMaximum();
			double max_mu = hist_temp_mu->GetMaximum();
			double global_max = max_e;
			if (max_mu > max_e) global_max = max_mu;

			TCanvas *c_comparison = new TCanvas(vector_varnames.at(i_var).c_str(),"Comparison Canvas",900,600);
            hist_temp_e->Draw("hist");
            hist_temp_mu->Draw("same hist");
			hist_temp_e->GetYaxis()->SetRangeUser(0,1.1*global_max);
            leg->Clear();
			leg->AddEntry(hist_temp_e,label1,"l");
            leg->AddEntry(hist_temp_mu,label2,"l");
            leg->Draw("same");
			vector_canvas.push_back(c_comparison);

			if (i_var==0){
				c_comparison->Print("ClassificationVars_Comparison_emu.pdf(","pdf");
			}
			else if (i_var == int(vector_varnames.size())-1){
				c_comparison->Print("ClassificationVars_Comparison_emu.pdf)","pdf");
			}
			else {
				c_comparison->Print("ClassificationVars_Comparison_emu.pdf","pdf");
			}
		}


		f_out->cd();
		for (unsigned int i_var = 0; i_var < vector_varnames.size(); i_var++){
			vector_canvas.at(i_var)->Write();
		}

		f_out->Close();
		f_e->Close();
		f_mu->Close();

		delete f_out;
		delete f_e;
		delete f_mu;


	}

}

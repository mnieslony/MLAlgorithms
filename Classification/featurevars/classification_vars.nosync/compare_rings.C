void compare_rings(const char* ringfile, const char* outfile, const char* configfile, const char* configtypefile, const char* label1="single-ring", const char* label2="multi-ring"){


	std::cout <<"compare_rings: Reading in file "<<ringfile<<std::endl;
	bool execute_program = true;


	ifstream file_ring(ringfile);
	if (!file_ring.good()) {
		std::cout <<"Specified ring file "<<ringfile<<" does not exist! Quit program..."<<std::endl;
		execute_program = false;
	}


	std::cout <<"Reading in variable configuration file "<<configfile<<std::endl;
	ifstream file_config(configfile);
	if (!file_config.good()) {
		std::cout <<"Specified variable configuration file "<<configfile<<" does not exist! Quit program..."<<std::endl;
		execute_program = false;
	}


	std::cout <<"Reading in variable configuration type file "<<configtypefile<<std::endl;
	ifstream file_configtype(configtypefile);
	if (!file_configtype.good()) {
		std::cout << "Specified variable configuration type file "<<configtypefile<<" does not exist! Quit program..."<<std::endl;
		execute_program = false;
	}



	if (execute_program){

		TFile *f_rings = new TFile(ringfile,"READ");
		TFile *f_out = new TFile(outfile,"RECREATE");

		std::vector<std::string> vector_varnames;
		std::string temp_string;
		while (!file_config.eof()){
			file_config >> temp_string;
			if (file_config.eof()) break;
			vector_varnames.push_back(temp_string);
		}
		file_config.close();
		
		std::map<std::string,int> map_vartypes;		
		std::map<std::string,int> map_nbins;
		std::map<std::string,double> map_minbin;
		std::map<std::string,double> map_maxbin;

		std::string temp_varname;
		int temp_nbins;
		double temp_minbin;
		double temp_maxbin;
		int temp_type;

		while (!file_configtype.eof()){
			file_configtype >> temp_varname >> temp_type >> temp_nbins >> temp_minbin >> temp_maxbin;
			if (file_configtype.eof()) break;
			map_vartypes.emplace(temp_varname,temp_type);
			map_nbins.emplace(temp_varname,temp_nbins);
			map_minbin.emplace(temp_varname,temp_minbin);
			map_maxbin.emplace(temp_varname,temp_maxbin);
		}
		file_configtype.close();


		TTree *classification_tree = (TTree*) f_rings->Get("classification_tree");
		int n_entries = classification_tree->GetEntries();

		std::map<std::string,int> variable_int;
		std::map<std::string,double> variable_double;
		std::map<std::string,bool> variable_bool;
		std::map<std::string,std::vector<double>*> variable_vector;

		bool fMCMultiRing;
		classification_tree->SetBranchAddress("MCMultiRing",&fMCMultiRing);

		for (int i=0; i<int(vector_varnames.size()); i++){

			std::string varname = vector_varnames.at(i);
			int vartype = map_vartypes[varname];
			if (vartype == 1) {
				variable_int.emplace(varname,-1);
				classification_tree->SetBranchAddress(varname.c_str(),&variable_int.at(varname));
			}
			else if (vartype ==2) {
				variable_double.emplace(varname,-1);
				classification_tree->SetBranchAddress(varname.c_str(),&variable_double.at(varname));
			}
			else if (vartype == 3){
				variable_bool.emplace(varname,0);
				classification_tree->SetBranchAddress(varname.c_str(),&variable_bool.at(varname));
			}
			else if (vartype == 4){
				std::vector<double> *temp_vector = new std::vector<double>;
				variable_vector.emplace(varname.c_str(),temp_vector);
				classification_tree->SetBranchAddress(varname.c_str(),&variable_vector.at(varname));
			}

		}

		TH1F *hist_temp_single = nullptr;
		TH1F *hist_temp_multi = nullptr;

		std::vector<TCanvas*> vector_canvas;
        TLegend *leg = new TLegend(0.75,0.75,0.9,0.9);
        std::map<std::string,TH1F*> histogram_map_single;
        std::map<std::string,TH1F*> histogram_map_multi;

		for (unsigned int i_var = 0; i_var < vector_varnames.size(); i_var++){

			std::string varname = vector_varnames.at(i_var);
			std::stringstream ss_histname_single, ss_histname_multi;
			ss_histname_single << "hist_"<<vector_varnames.at(i_var) <<"_single";
			ss_histname_multi << "hist_"<<vector_varnames.at(i_var) <<"_multi";
			TH1F *hist_temp_single = new TH1F(ss_histname_single.str().c_str(),ss_histname_single.str().c_str(),map_nbins[varname],map_minbin[varname],map_maxbin[varname]);
			TH1F *hist_temp_multi = new TH1F(ss_histname_multi.str().c_str(),ss_histname_multi.str().c_str(),map_nbins[varname],map_minbin[varname],map_maxbin[varname]);

			histogram_map_single.emplace(vector_varnames.at(i_var),hist_temp_single);
			histogram_map_multi.emplace(vector_varnames.at(i_var),hist_temp_multi);

		}

		for (int i_entry=0; i_entry < n_entries; i_entry++) {

			classification_tree->GetEntry(i_entry);
			bool temp_multiring = variable_bool["MCMultiRing"];
			for (int i_var=0; i_var < int(vector_varnames.size()); i_var++){
				std::string varname = vector_varnames.at(i_var);
				int vartype = map_vartypes[varname];
				if (vartype == 1){
					int variable = variable_int[varname];
					if (temp_multiring) histogram_map_multi[varname]->Fill(variable);
					else histogram_map_single[varname]->Fill(variable);
				}
				else if (vartype == 2){
					double variable = variable_double[varname];
					if (temp_multiring) histogram_map_multi[varname]->Fill(variable);
					else histogram_map_single[varname]->Fill(variable);
				}
				else if (vartype == 3){
					bool variable = variable_bool[varname];
					if (temp_multiring) histogram_map_multi[varname]->Fill(variable);
					else histogram_map_single[varname]->Fill(variable);
				}
				else if (vartype == 4){
					std::vector<double>* variable = variable_vector[varname];
					for (int j=0; j < int(variable->size()); j++){
						double single_variable = variable->at(j);
						if (temp_multiring) histogram_map_multi[varname]->Fill(single_variable);
						else histogram_map_single[varname]->Fill(single_variable);
					}
				}
			}
		}

		for (int i_var = 0; i_var < int(vector_varnames.size()); i_var++){

			std::string variable = vector_varnames.at(i_var);
			TH1F *hist_single = histogram_map_single[variable];
			TH1F *hist_multi = histogram_map_multi[variable];
			hist_single->Scale(1./hist_single->Integral());
			hist_multi->Scale(1./hist_multi->Integral());
			hist_single->SetLineColor(2);
			hist_single->SetStats(0);
			double max_single = hist_single->GetMaximum();
			double max_multi = hist_multi->GetMaximum();
			double global_max = max_single;
			if (max_multi > max_single) global_max = max_multi;

			TCanvas *c_comparison = new TCanvas(variable.c_str(),"Comparison Canvas",900,600);
			hist_single->Draw("hist");
			hist_multi->Draw("same hist");
			hist_single->GetYaxis()->SetRangeUser(0,1.1*global_max);
			leg->Clear();
			leg->AddEntry(hist_single,label1,"l");
			leg->AddEntry(hist_multi,label2,"l");
			leg->Draw("same");
			vector_canvas.push_back(c_comparison);

			if (i_var==0){
				c_comparison->Print("ClassificationVars_Comparison_rings.pdf(","pdf");
			}
			else if (i_var == int(vector_varnames.size())-1){
				c_comparison->Print("ClassificationVars_Comparison_rings.pdf)","pdf");
			}
			else {
				c_comparison->Print("ClassificationVars_Comparison_rings.pdf","pdf");
			}
		}


		f_out->cd();
		for (unsigned int i_var = 0; i_var < vector_varnames.size(); i_var++){
			vector_canvas.at(i_var)->Write();
		}

		f_out->Close();
		f_rings->Close();

		delete f_out;
		delete f_rings;

	}

}

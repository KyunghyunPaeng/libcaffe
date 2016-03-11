import os, re, time
import yaml
import caffe
import subprocess
import ConfigParser

class Trainer(object):
    def __init__(self, conf_file):
        """
            [conf_file]
            This class manages all processes (data preparation, model training, model testing)
            based on input configuration file
            'workspace' means directory consisting of data and training trials...
        """
        self._conf = self._parse_config(conf_file)
        self._create_directory()
        self._make_prototxts()

    @property
    def workspace_path(self):
        return self._conf['general']['workspace_path']
    @property
    def engine_path(self):
        return self._conf['general']['engine_path']
    @property
    def base_pt_path(self):
        return os.path.join(self.workspace_path,
               'base_prototxt', self._conf['trainer']['trn_base_prototxt'])
    @property
    def current_pt_path(self):
        return os.path.join(self.workspace_path,'trial',
                            self.trial_path, '_prototxts')
    @property
    def snapshot_path(self):
        return os.path.join(self.workspace_path,'trial',
                            self.trial_path, '_snapshots')
    @property
    def solver_path(self):
        return os.path.join(self.current_pt_path,'solver.pt')
    @property
    def log_path(self):
        return os.path.join(self.workspace_path,'trial',
                            self.trial_path)
    @property
    def trial_path(self):
        return self._conf['trainer']['trial_id']

    def training(self, gpu_id=0):
        CAFFE_COMMAND = os.path.join(self.engine_path, 'build/tools/caffe')
        if type(gpu_id) is list:
            gpu_id = [str(i) for i in gpu_id]
            gpu_id = ','.join(gpu_id)
        procs = []
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        sol_proto_log = timestamp + '.log'
        with open(os.path.join(self.log_path, sol_proto_log), 'a') as f:
            shellLine = [CAFFE_COMMAND, 'train', '--solver='+self.solver_path, '--gpu='+str(gpu_id)] 
            p = subprocess.Popen(shellLine, stdout=f, stderr=f)
            procs.append(p)
        return procs
    
    def fine_tuning(self, gpu_id=0):
        CAFFE_COMMAND = os.path.join(self.engine_path, 'build/tools/caffe')
        weight_path = self._conf['fine_tuning']['weights']
        if type(gpu_id) is list:
            gpu_id = [str(i) for i in gpu_id]
            gpu_id = ','.join(gpu_id)
        procs = []
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        sol_proto_log = timestamp + '.log'
        with open(os.path.join(self.log_path, sol_proto_log), 'a') as f:
            shellLine = [CAFFE_COMMAND, 'train', '--solver='+self.solver_path, '--weights='+weight_path, '--gpu='+str(gpu_id)] 
            p = subprocess.Popen(shellLine, stdout=f, stderr=f)
            procs.append(p)
        return procs
           
    def _make_prototxts(self):
        #TODO: Need deploy file generation, repeated field control (mean_value)
        self._make_train_pt()
        self._make_solver_pt()
        #self._make_deploy_pt()

    def _make_solver_pt(self):
        solver = caffe.io.caffe_pb2.SolverParameter()
        for item in self._conf['solver'].iteritems():
            value = yaml.load(item[1])
            if value in solver.DESCRIPTOR.enum_values_by_name.keys(): # if enum variable
                value = solver.DESCRIPTOR.enum_values_by_name[value].number
            try:
                setattr(solver, item[0], value)
            except: # for repeated field !
                tmp = getattr(solver, item[0])
                while len(tmp) != 0:
                    del tmp[len(tmp)-1]
                tmp.append(value)
        solver.net = os.path.join(self.current_pt_path, 'train_val.pt')
        solver.snapshot_prefix = self.snapshot_path + '/model'
        save_solver_path = os.path.join(self.current_pt_path, 'solver.pt')
        open(save_solver_path,'w').write(str(solver))

    def _make_train_pt(self):
        from google.protobuf import text_format
        model = caffe.io.caffe_pb2.NetParameter()
        text_format.Merge(open(self.base_pt_path).read(), model)
        for layer in model.layer:
            if layer.name == 'data':
                if layer.include[0].phase==0: 
                    conf = self._conf['trainer']
                else: 
                    conf = self._conf['tester']
                # transform param
                target = layer.transform_param
                for item in conf.iteritems():
                    try:
                        setattr(target, item[0], yaml.load(item[1]))
                    except:
                        try: # repeated field
                            tmp = getattr(target, item[0])
                            while len(tmp) != 0:
                               del tmp[len(tmp)-1]
                            tmp.append(yaml.load(item[1]))
                        except:
                            pass
                # other param
                target = self._get_target_param(layer)
                for item in conf.iteritems():
                    try:
                        setattr(target, item[0], yaml.load(item[1]))
                    except:
                        try:
                            tmp = getattr(target, item[0])
                            while len(tmp) != 0:
                               del tmp[len(tmp)-1]
                            tmp.append(yaml.load(item[1]))
                        except:
                            pass
        save_pt_path = os.path.join(self.current_pt_path,'train_val.pt')
        open(save_pt_path,'w').write(str(model))
    
    def _get_target_param(self, layer):
        params = ''
        parsing_str = re.findall('[A-Z][^A-Z]*', str(layer.type) )
        for s in parsing_str:
            params += s.lower() + '_'
        params += 'param'
        try:
            target = getattr(layer, params)
        except:
            target = None
        return target

    def _create_directory(self):
        if not os.path.exists(self.base_pt_path):
            assert False, 'Cannot find {}'.format(self.base_pt_path)
        w_path = os.path.join(self.workspace_path,'trial')
        t_path = os.path.join(self.workspace_path,'trial', self.trial_path)
        p_path = os.path.join(t_path, '_prototxts')
        s_path = os.path.join(t_path, '_snapshots')
        if not os.path.exists(w_path): os.mkdir(w_path)
        if not os.path.exists(t_path): os.mkdir(t_path)
        if not os.path.exists(p_path): os.mkdir(p_path)
        if not os.path.exists(s_path): os.mkdir(s_path)

    def _parse_config(self, conf):
        config = ConfigParser.ConfigParser()
        config.read(conf)
        KEY_TRAINING = ['trainer','tester','solver']
        config_dict = {s: {k:v for k,v in config.items(s)} for s in config.sections()}
        if 'fine_tuning' in config_dict.keys():
            KEY_TRAINING.append('fine_tuning')
        if 'resume' in config_dict.keys():
            KEY_TRAINING.append('resume')
        config_general = {k:config_dict[k] for k in config_dict if k == 'general'}
        config_training = {k:config_dict[k] for k in config_dict if k in KEY_TRAINING}
        workspace = config_dict['general']['workspace_path']
        if not os.path.exists(workspace): os.mkdir(workspace)
        training_config = config_general.copy()
        training_config.update(config_training)
        return training_config


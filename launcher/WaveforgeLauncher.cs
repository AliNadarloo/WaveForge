// Made by Ali Nadarloo.
// SPDX-License-Identifier: MIT
// Lightweight Windows launcher. Python and numerical libraries are external.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Web.Script.Serialization;
using System.Windows.Forms;

static class WaveforgeLauncher {
    static string Home=Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
    static JavaScriptSerializer Json=new JavaScriptSerializer();

    [STAThread] static int Main(string[] args) {
        bool setup=Array.IndexOf(args,"--setup")>=0;
        bool cli=Array.IndexOf(args,"--headless")>=0 || Array.IndexOf(args,"--smoke-gui")>=0;
        Application.EnableVisualStyles();
        try {
            ExtractPayload();string python=FindPython();
            if(setup || python==null) {
                if(cli && !setup) { File.WriteAllText(Path.Combine(Home,"waveforge_error.log"),"No compatible Python found. Run WaveForge.exe --setup.");return 2; }
                using(SetupForm form=new SetupForm(python)) { if(form.ShowDialog()!=DialogResult.OK)return 0;python=form.PythonPath; }
            }
            File.WriteAllText(Path.Combine(Home,"launcher_settings.json"),Json.Serialize(new Dictionary<string,string>{{"python",python}}));
            List<string> command=new List<string>{Path.Combine(Home,"app_source","main.py")};
            foreach(string arg in args)if(arg!="--setup")command.Add(arg);
            List<string> quoted=new List<string>();foreach(string value in command)quoted.Add(Quote(value));
            ProcessStartInfo start=Info(python,String.Join(" ",quoted.ToArray()));
            start.EnvironmentVariables["WAVEFORGE_HOME"]=Home;start.EnvironmentVariables["PYTHONDONTWRITEBYTECODE"]="1";
            start.EnvironmentVariables["MPLBACKEND"]="Agg";
            using(Process child=Process.Start(start)) {
                StringBuilder errors=new StringBuilder();
                child.OutputDataReceived+=delegate(object s,DataReceivedEventArgs e){};
                child.ErrorDataReceived+=delegate(object s,DataReceivedEventArgs e){if(e.Data!=null)lock(errors){if(errors.Length<100000)errors.AppendLine(e.Data);}};
                child.BeginOutputReadLine();child.BeginErrorReadLine();child.WaitForExit();
                if(child.ExitCode!=0 && errors.Length>0)File.WriteAllText(Path.Combine(Home,"launcher_error.log"),errors.ToString());
                if(child.ExitCode!=0 && !cli)MessageBox.Show("The app stopped. See waveforge_error.log or launcher_error.log beside the executable. Reopen setup with WaveForge.exe --setup.","WaveForge");
                return child.ExitCode;
            }
        } catch(Exception error) {
            try{File.WriteAllText(Path.Combine(Home,"launcher_error.log"),error.ToString());}catch{}
            if(!cli)MessageBox.Show(error.Message+"\nKeep the app in a writable folder. See INSTALL.md.","WaveForge setup");return 1;
        }
    }

    static string Hash(byte[] data){using(SHA256 h=SHA256.Create())return BitConverter.ToString(h.ComputeHash(data)).Replace("-","").ToLowerInvariant();}
    static void ExtractPayload() {
        string state=Path.Combine(Home,"app_source","distribution.json");Dictionary<string,string> old=new Dictionary<string,string>();
        try{old=Json.Deserialize<Dictionary<string,string>>(File.ReadAllText(state));}catch{}
        Dictionary<string,string> next=new Dictionary<string,string>();
        using(Stream stream=Assembly.GetExecutingAssembly().GetManifestResourceStream("Waveforge.payload.zip"))
        using(ZipArchive archive=new ZipArchive(stream,ZipArchiveMode.Read))foreach(ZipArchiveEntry entry in archive.Entries) {
            string target=Path.GetFullPath(Path.Combine(Home,entry.FullName.Replace('/',Path.DirectorySeparatorChar)));
            if(!target.StartsWith(Home+Path.DirectorySeparatorChar,StringComparison.OrdinalIgnoreCase))throw new IOException("Invalid embedded path.");
            byte[] data;using(Stream source=entry.Open())using(MemoryStream memory=new MemoryStream()){source.CopyTo(memory);data=memory.ToArray();}
            string expected=Hash(data);next[entry.FullName]=expected;bool write=!File.Exists(target);
            if(!write) { string current=Hash(File.ReadAllBytes(target));string previous;
                if(current==expected)continue;
                write=old.TryGetValue(entry.FullName,out previous) && current==previous;
            }
            // A file edited by the user is preserved across app updates.
            if(write){Directory.CreateDirectory(Path.GetDirectoryName(target));File.WriteAllBytes(target,data);}
        }
        Directory.CreateDirectory(Path.GetDirectoryName(state));File.WriteAllText(state,Json.Serialize(next));
    }

    static ProcessStartInfo Info(string file,string args){return new ProcessStartInfo(file,args){WorkingDirectory=Home,UseShellExecute=false,CreateNoWindow=true,RedirectStandardOutput=true,RedirectStandardError=true};}
    static string Probe(string file,string args) {
        try{using(Process p=Process.Start(Info(file,args))){if(!p.WaitForExit(10000)){p.Kill();return null;}string result=p.StandardOutput.ReadToEnd().Trim();return p.ExitCode==0?result:null;}}catch{return null;}
    }
    static bool Ready(string python) {
        if(String.IsNullOrWhiteSpace(python)||!File.Exists(python))return false;
        if(python.IndexOf("\\WindowsApps\\",StringComparison.OrdinalIgnoreCase)>=0)return false;
        string code="import sys,importlib.util;sys.exit(0 if sys.version_info>=(3,10) and all(importlib.util.find_spec(m) for m in ['numpy','torch','matplotlib','PIL','PySide6']) else 1)";
        return Probe(python,"-c "+Quote(code))!=null;
    }
    static string FindPython() {
        List<string> choices=new List<string>();
        try{choices.Add(Json.Deserialize<Dictionary<string,string>>(File.ReadAllText(Path.Combine(Home,"launcher_settings.json")))["python"]);}catch{}
        choices.Add(Environment.GetEnvironmentVariable("WAVEFORGE_PYTHON"));choices.Add(Path.Combine(Home,".venv","Scripts","python.exe"));
        foreach(string path in choices)if(Ready(path))return path;
        string py=Probe("py.exe","-3 -c "+Quote("import sys;print(sys.executable)"));if(Ready(py))return py;
        string paths=Probe("where.exe","python.exe");
        if(paths!=null)foreach(string path in paths.Split(new char[]{'\r','\n'},StringSplitOptions.RemoveEmptyEntries))if(Ready(path))return path;
        string local=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),"Programs","Python");
        if(Directory.Exists(local))foreach(string dir in Directory.GetDirectories(local)){string path=Path.Combine(dir,"python.exe");if(Ready(path))return path;}
        return null;
    }
    // Windows argument escaping, including quotes and trailing backslashes.
    static string Quote(string value) {
        StringBuilder text=new StringBuilder("\"");int slashes=0;
        foreach(char ch in value){if(ch=='\\'){slashes++;continue;}if(ch=='"'){text.Append('\\',slashes*2+1);text.Append(ch);slashes=0;continue;}text.Append('\\',slashes);slashes=0;text.Append(ch);}
        text.Append('\\',slashes*2);text.Append('"');return text.ToString();
    }
    static string PSQuote(string value){return "'"+value.Replace("'","''")+"'";}

    class SetupForm:Form {
        public string PythonPath;TextBox path;Label status;Timer capture;
        public SetupForm(string python) {
            Text="WaveForge / Python setup — Made by Ali Nadarloo";Size=new Size(820,720);MinimumSize=new Size(720,640);StartPosition=FormStartPosition.CenterScreen;
            BackColor=Color.FromArgb(9,15,27);ForeColor=Color.FromArgb(220,230,243);Font=new Font("Segoe UI",10);
            TableLayoutPanel layout=new TableLayoutPanel{Dock=DockStyle.Fill,Padding=new Padding(24),ColumnCount=1,RowCount=6};Controls.Add(layout);
            layout.Controls.Add(new Label{Text="WAVEFORGE / LIGHTWEIGHT LAUNCHER",AutoSize=true,ForeColor=Color.Aquamarine,Font=new Font("Segoe UI",15,FontStyle.Bold)});
            layout.Controls.Add(new Label{Text="Choose Python with PyTorch, NumPy, Matplotlib, Pillow and PySide6. This launcher does not bundle or download CUDA libraries.",AutoSize=true,MaximumSize=new Size(740,0)});
            FlowLayoutPanel row=new FlowLayoutPanel{AutoSize=true,Dock=DockStyle.Fill};path=new TextBox{Width=550,Text=python??""};row.Controls.Add(path);
            Button browse=Button("Browse…");browse.Click+=delegate{using(OpenFileDialog dialog=new OpenFileDialog{Filter="Python executable|python.exe|Executables|*.exe",CheckFileExists=true})if(dialog.ShowDialog()==DialogResult.OK)path.Text=dialog.FileName;};row.Controls.Add(browse);layout.Controls.Add(row);
            status=new Label{Text="Use the same Python environment used to install the packages. The guide below includes complete setup steps.",AutoSize=true,MaximumSize=new Size(740,0)};layout.Controls.Add(status);
            TextBox guide=new TextBox{Multiline=true,ReadOnly=true,ScrollBars=ScrollBars.Vertical,Dock=DockStyle.Fill,Font=new Font("Consolas",10),BackColor=Color.FromArgb(17,28,45),ForeColor=ForeColor,Text=File.ReadAllText(Path.Combine(Home,"INSTALL.md")).Replace("\r\n","\n").Replace("\n",Environment.NewLine)};layout.Controls.Add(guide);
            for(int i=0;i<4;i++)layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));layout.RowStyles.Add(new RowStyle(SizeType.Percent,100));layout.RowStyles.Add(new RowStyle(SizeType.AutoSize));
            FlowLayoutPanel actions=new FlowLayoutPanel{AutoSize=true,Dock=DockStyle.Fill};
            foreach(string kind in new string[]{"CPU","GPU"}){string selected=kind;Button copy=Button("Copy "+kind+" commands");copy.Click+=delegate{string exe=String.IsNullOrWhiteSpace(path.Text)?".venv\\Scripts\\python.exe":path.Text;Clipboard.SetText("& "+PSQuote(exe)+" -m pip install -r "+PSQuote(Path.Combine(Home,"requirements-runtime.txt"))+Environment.NewLine+"& "+PSQuote(exe)+" -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/"+(selected=="GPU"?"cu121":"cpu"));status.Text="Commands copied. Run in PowerShell, then Check and start.";};actions.Controls.Add(copy);}
            Button run=Button("Check and start");run.Click+=delegate{status.Text="Checking Python…";status.Refresh();if(Ready(path.Text.Trim())){PythonPath=path.Text.Trim();DialogResult=DialogResult.OK;Close();}else status.Text="This Python is missing packages or is older than 3.10. Follow the guide and try again.";};actions.Controls.Add(run);layout.Controls.Add(actions);
            string screenshot=Environment.GetEnvironmentVariable("WAVEFORGE_SETUP_SCREENSHOT");
            if(screenshot!=null){capture=new Timer{Interval=600};capture.Tick+=delegate{capture.Stop();using(Bitmap image=new Bitmap(Width,Height)){DrawToBitmap(image,new Rectangle(0,0,Width,Height));image.Save(screenshot);}DialogResult=DialogResult.Cancel;Close();};Shown+=delegate{capture.Start();};}
        }
        Button Button(string text){return new Button{Text=text,AutoSize=true,BackColor=Color.FromArgb(20,34,56),ForeColor=Color.Aquamarine,FlatStyle=FlatStyle.Flat};}
        protected override void Dispose(bool disposing){if(disposing&&capture!=null)capture.Dispose();base.Dispose(disposing);}
    }
}

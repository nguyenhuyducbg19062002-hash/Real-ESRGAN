; Trinh cai dat "Video 4K Noel"
; Build: ISCC.exe installer_src\Video4KNoel.iss   (sau khi da chay PyInstaller ra release\Video4KNoel)

#define AppName "Video 4K Noel"
#define AppVersion "1.2.0"
#define AppExe "Video4KNoel.exe"

[Setup]
AppId={{6E2B7C1A-4F0D-4E7B-9C7E-2A9F3D1B4C58}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppName}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Cai cho nguoi dung hien tai, khong can quyen Admin (co the chon cai cho moi nguoi)
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\installer
OutputBaseFilename=Video4KNoel_Setup_{#AppVersion}
SetupIconFile=..\app_assets\icon.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
; App dang mo (mutex trong app_4k.py) -> trinh cai/go se bao va de nghi dong app truoc
AppMutex=Video4KNoel_SingleInstance
CloseApplications=yes
WizardStyle=modern
WizardImageFile=wizard_100.bmp,wizard_200.bmp
WizardSmallImageFile=wizard_small_100.bmp,wizard_small_200.bmp
Compression=lzma2/fast
SolidCompression=no
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=8
; Ban dong goi > 2GB -> chia thanh Setup.exe + cac file .bin di kem (de cung thu muc khi cai)
DiskSpanning=yes
DiskSliceSize=max
ExtraDiskSpaceRequired=104857600

[Languages]
Name: "vi"; MessagesFile: "Vietnamese.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
vi.NoNvidia=Không tìm thấy card đồ họa NVIDIA (hoặc driver NVIDIA) trên máy này.%n%nApp vẫn chạy được nhưng sẽ dùng CPU nên RẤT chậm. Nên cài driver NVIDIA mới nhất trước.%n%nVẫn tiếp tục cài đặt?
en.NoNvidia=No NVIDIA GPU/driver was found.%n%nThe app will fall back to the CPU and be VERY slow. Installing the latest NVIDIA driver first is recommended.%n%nContinue anyway?
vi.LaunchApp=Mở {#AppName} ngay
en.LaunchApp=Launch {#AppName} now

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "..\release\Video4KNoel\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[UninstallDelete]
; Cache TensorRT + du lieu tam cua app. KHONG xoa video da xuat (Videos\Video 4K).
Type: filesandordirs; Name: "{localappdata}\Video4KNoel"
Type: filesandordirs; Name: "{app}"

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchApp}"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
  if not FileExists(ExpandConstant('{sys}\nvidia-smi.exe')) then
    Result := MsgBox(CustomMessage('NoNvidia'), mbConfirmation, MB_YESNO) = IDYES;
end;

# API 安全性指引

此專案已不再將 API Key 存放於任何 `appsettings` 檔案，請改用 [使用者祕密](https://learn.microsoft.com/aspnet/core/security/app-secrets)、環境變數或 Azure Key Vault 等外部祕密管理服務提供設定值。

## 1. 在本機設定 user secrets
1. 確認 `F3.ToolHub.csproj` 已包含 `UserSecretsId`（此版本已內建）。
2. 在每台開發機執行下列指令：
   ```bash
   cd D:/Project/F3.ToolHub
   dotnet user-secrets set "ApiSecurity:ApiKeys:0:Name" "local-dev"
   dotnet user-secrets set "ApiSecurity:ApiKeys:0:Key" "<your-random-key>"
   dotnet user-secrets set "ApiSecurity:ApiKeys:0:Roles:0" "tools.read"
   dotnet user-secrets set "ApiSecurity:ApiKeys:0:Roles:1" "tools.execute"
   ```
3. 重新啟動 API。所有請求都必須帶上 `ApiSecurity:HeaderName` 指定的標頭（預設為 `X-Api-Key`）。

## 2. 正式／預備環境
- 使用部署平台提供的祕密儲存機制（如 Azure App Service 設定、Kubernetes Secret、AWS Parameter Store）填入相同的設定鍵。
- 勿將正式金鑰寫入映像檔或版控中的設定檔。
- 定期輪替金鑰，並透過既有的執行紀錄追蹤其使用情況。

## 3. 測試情境覆寫
整合測試與模擬器可以在啟動測試 Host 時，以記憶體組態注入臨時金鑰；可參考 `F3.ToolHub.IntegrationTests` 中覆寫 `ApiSecurity` 的示範。

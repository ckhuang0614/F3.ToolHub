# F3.ToolHub

F3.ToolHub 提供一組以 Modbus TCP 為主的工具 API，可透過 `X-Api-Key` 驗證存取 PLC 讀取功能。請搭配 `appsettings.*.json` 與使用者祕密設定對應的 Modbus 與 API Key 參數。

- 服務已改用 Serilog；調整 `Serilog` 設定節即可切換 console/檔案/集中式收集等寫入管道。

## 專案結構
- `F3.ToolHub`：主要 Web API（.NET 10 Minimal API），包含 Modbus 工具、背景輪詢、API Key 驗證等元件。
- `F3.ToolHub.IntegrationTests`：整合測試專案，使用 xUnit、`Microsoft.AspNetCore.TestHost` 以及內建 `ModbusSimulator` 驗證端對端流程。

## Modbus 暫存器對應
- 透過 `Modbus:Actions[n]:Registers` 描述每個邏輯點：`Name`、`Offset`、`DataType`（`UInt16`/`Int16`/`UInt32`/`Int32`/`Float32`/`Boolean`）、`Scale` 與 `Unit`/`BitIndex`，並可選填 `MinValue`、`MaxValue` 產生品質旗標。
- 執行結果與 `plc/data` 端點會同時輸出原始暫存器與轉換後的 `Points`（含 `Quality`）。
- `/api/tools/plc/mappings` 端點允許 CRUD 映射，方便 UI/表格存取，即時調整 action 與暫存器對應。
- `Output` 模組可同時寫入 JSONL 歷史檔與 MQTT broker，亦可自行新增 `IPlcDataSink` 實作串接資料庫 / OPC UA。
- `Monitoring.AlertWebhooks` 可設定多個 Webhook，服務會於 `PlcAlertService` 內同步推播品質與連線異常，方便串接 Teams/Slack/值班系統。
- `plc/alerts` 與 `/healthz` 可檢視品質告警與 Modbus 連線健康度，亦可整合監控平台觸發通知。
- `plc/health` 端點可查詢 Modbus 連線池狀態、成功/失敗次數與熔斷資訊，以利監控。

## 整合測試
整合測試專案 `F3.ToolHub.IntegrationTests` 目前涵蓋三個情境：
1. `MissingApiKey_ShouldReturn401`：在無 API Key 時呼叫 `/api/tools`，應收到 401，驗證認證中介層。
2. `ListTools_ShouldExposeModbusTool`：帶 API Key 呼叫 `/api/tools`，確保 `ModbusPlcToolProvider.ToolId` 有被公開，驗證工具登錄與用例。
3. `ExecuteModbusAction_ShouldReturnSnapshot`：啟動 `ModbusSimulator`、呼叫 `/api/tools/{toolId}/actions/demo-holding-registers`，再讀取 `/api/tools/plc/data`，確認 Modbus TCP 客戶端、快取與背景輪詢能完整運作。

### 執行方式
在方案根目錄執行：

```bash
dotnet test .\F3.ToolHub.IntegrationTests\F3.ToolHub.IntegrationTests.csproj
```

測試會自動啟動模擬器、覆寫設定並以 `TestServer` 啟動 API，無需額外依賴 PLC 硬體。

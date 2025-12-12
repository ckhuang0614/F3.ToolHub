using System.Collections.Generic;
using F3.ToolHub.Domain.Monitoring;

namespace F3.ToolHub.Application.Contracts;

public interface IPlcAlertService
{
    void Raise(PlcAlert alert);

    IReadOnlyCollection<PlcAlert> GetRecent(int? take = null);
}

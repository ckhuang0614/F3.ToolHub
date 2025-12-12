using System.Threading;
using System.Threading.Tasks;
using F3.ToolHub.Domain.Monitoring;

namespace F3.ToolHub.Application.Contracts;

public interface IPlcAlertSink
{
    Task PublishAsync(PlcAlert alert, CancellationToken cancellationToken);
}

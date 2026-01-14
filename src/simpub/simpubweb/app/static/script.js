document.addEventListener('DOMContentLoaded', () => {
    const scanButton = document.getElementById('scan-button');
    const xrTemplate = document.getElementById('xr-template');
    const deviceContainer = document.getElementById('device-container');

    scanButton.addEventListener('click', () => {
        fetch('/scan')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Failed to scan devices');
                }
                return response.json();
            })
            .then(responseData => {
                if (responseData.status !== 'success') {
                    throw new Error(responseData.message || 'Scan failed');
                }
                deviceContainer.innerHTML = '';
                const nodes = Array.isArray(responseData.nodes) ? responseData.nodes : [];
                nodes.forEach(node => addNodePanel(node));
            })
            .catch(error => {
                alert(`${error.message}`);
            });
    });

    function addNodePanel(nodeInfo) {
        const clone = xrTemplate.content.cloneNode(true);
        const deviceHeader = clone.querySelector('.device-header');
        const ip = nodeInfo.ip || (nodeInfo.addr && nodeInfo.addr.ip) || '';
        const port = nodeInfo.port ?? '';

        deviceHeader.setAttribute('data-name', nodeInfo.name || '');
        deviceHeader.setAttribute('data-node-id', nodeInfo.nodeID || '');
        deviceHeader.setAttribute('data-ip', ip);
        deviceHeader.setAttribute('data-port', port);

        clone.querySelector('.device-name').textContent = nodeInfo.name || 'Unknown Node';

        const typeEl = clone.querySelector('.device-type');
        typeEl.textContent = nodeInfo.type || 'Unknown';

        const ipEl = clone.querySelector('.device-ip');
        ipEl.textContent = ip ? (port ? `${ip}:${port}` : ip) : 'Unavailable';

        const nodeIdEl = clone.querySelector('.device-id');
        nodeIdEl.textContent = nodeInfo.nodeID || 'Unavailable';

        addButtonEventListeners(clone, deviceHeader);
        deviceContainer.appendChild(clone);
    }

    function addButtonEventListeners(clone, deviceHeader) {
        const name = deviceHeader.getAttribute('data-name');
        const ip = deviceHeader.getAttribute('data-ip');
        const port = deviceHeader.getAttribute('data-port');

        // Teleport
        clone.querySelector('.teleport-btn').addEventListener('click', () => {
            sendPostRequest('/teleport-scene', { name, ip, servicePort: port }, 'Teleport Scene');
        });

        // Rename
        clone.querySelector('.rename-btn').addEventListener('click', () => {
            const newName = prompt('Enter new name for the device:');
            if (!newName) {
                return;
            }
            sendPostRequest('/rename-device', { name, ip, servicePort: port, newName }, 'Rename Device');
            alert(`Device renamed to: ${newName}`);
        });

        // Environment Occlusion
        clone.querySelector('.env-occlusion-btn').addEventListener('click', () => {
            sendPostRequest('/env-occlusion', { name, ip, servicePort: port }, 'Environment Occlusion');
        });

        // Toggle QR-Tracking
        const toggleBtn = clone.querySelector('.toggle-qr-btn');
        toggleBtn.addEventListener('click', () => {
            // Use direct fetch here so we don't need to modify sendPostRequest
            fetch('/toggle-qr-tracking', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, ip, servicePort: port }),
            })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Request failed');
                    }
                    return response.json();
                })
                .then(result => {
                    if (result && result.status === 'success') {
                        console.log('Toggle QR Tracking succeeded, message:', result.message);
                        toggleBtn.classList.toggle('active');
                    } else {
                        alert(`Toggle QR Tracking failed: ${result?.message || 'Unknown error'}`);
                    }
                })
                .catch(error => {
                    console.error('Toggle QR Tracking Error:', error);
                    alert('Toggle QR Tracking failed.');
                });
        });

        // Log
        clone.querySelector('.log-btn').addEventListener('click', () => {
            alert(`Log requested for Node: ${deviceHeader.getAttribute('data-node-id')}`);
        });
    }

    function sendPostRequest(url, body, actionName) {
        fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        })
            .then(response => response.json())
            .then(result => {
                if (result.status !== 'success') {
                    alert(`${actionName} failed: ${result.message || 'Unknown error'}`);
                }
            })
            .catch(error => {
                console.error(`${actionName} Error:`, error);
                alert(`${actionName} failed.`);
            });
    }
});

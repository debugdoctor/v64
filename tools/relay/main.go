// A WISP server for v64's `wisp://` network backend.
//
// WISP carries only TCP/UDP payloads, so this program has no DHCP, DNS or NAT:
// the guest-facing network stack (DHCP, DNS, ARP, ICMP) lives in the client,
// fake_network.js. All this has to do is accept a stream id, dial the target
// and pump bytes both ways. That keeps it unprivileged and portable: no TAP
// device, no dnsmasq, no firewall rules, and one static binary per platform.
//
// Framing (little-endian), from src/browser/wisp_network.js:
//
//	[u8 type][u32 stream_id][payload]
//
//	0x01 CONNECT  payload: [u8 stream_type=1][u16 port][hostname bytes]
//	0x02 DATA     payload: bytes
//	0x03 CONTINUE payload: [u32 credit]
//	0x04 CLOSE    payload: [u8 reason]
//
// The server must send a CONTINUE for stream 0 immediately after the handshake.
// The client starts every stream with zero credit and buffers everything --
// including its own CONNECT -- until one arrives, so without it nothing is ever
// sent.
package main

import (
	"bufio"
	"crypto/sha1"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
)

const (
	frameConnect  = 0x01
	frameData     = 0x02
	frameContinue = 0x03
	frameClose    = 0x04

	streamTCP = 1

	// Credit granted per stream. Each DATA the client sends spends one; when it
	// runs out the client buffers until the next CONTINUE.
	creditWindow = 256

	closeVoluntary = 0x02
	closeNetwork   = 0x03

	dialTimeout = 30 * time.Second
)

// ---------------------------------------------------------------------------
// Minimal RFC 6455 server-side WebSocket. The standard library has none, and
// pulling a module would make the build depend on a reachable module proxy.
// ---------------------------------------------------------------------------

const wsGUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

func acceptKey(key string) string {
	sum := sha1.Sum([]byte(key + wsGUID))
	return base64.StdEncoding.EncodeToString(sum[:])
}

type wsConn struct {
	conn net.Conn
	r    *bufio.Reader
	mu   sync.Mutex // serialises writes
}

func upgrade(w http.ResponseWriter, r *http.Request) (*wsConn, error) {
	if !strings.EqualFold(r.Header.Get("Upgrade"), "websocket") {
		return nil, errors.New("not a websocket upgrade")
	}
	key := r.Header.Get("Sec-WebSocket-Key")
	if key == "" {
		return nil, errors.New("missing Sec-WebSocket-Key")
	}

	hijacker, ok := w.(http.Hijacker)
	if !ok {
		return nil, errors.New("response writer cannot be hijacked")
	}
	conn, buf, err := hijacker.Hijack()
	if err != nil {
		return nil, err
	}

	response := "HTTP/1.1 101 Switching Protocols\r\n" +
		"Upgrade: websocket\r\n" +
		"Connection: Upgrade\r\n" +
		"Sec-WebSocket-Accept: " + acceptKey(key) + "\r\n\r\n"
	if _, err := buf.WriteString(response); err != nil {
		conn.Close()
		return nil, err
	}
	if err := buf.Flush(); err != nil {
		conn.Close()
		return nil, err
	}

	return &wsConn{conn: conn, r: buf.Reader}, nil
}

// readMessage returns the next binary message, reassembling fragments.
func (c *wsConn) readMessage() ([]byte, error) {
	var message []byte

	for {
		var header [2]byte
		if _, err := io.ReadFull(c.r, header[:]); err != nil {
			return nil, err
		}

		fin := header[0]&0x80 != 0
		opcode := header[0] & 0x0F
		masked := header[1]&0x80 != 0
		length := uint64(header[1] & 0x7F)

		switch length {
		case 126:
			var ext [2]byte
			if _, err := io.ReadFull(c.r, ext[:]); err != nil {
				return nil, err
			}
			length = uint64(binary.BigEndian.Uint16(ext[:]))
		case 127:
			var ext [8]byte
			if _, err := io.ReadFull(c.r, ext[:]); err != nil {
				return nil, err
			}
			length = binary.BigEndian.Uint64(ext[:])
		}

		var mask [4]byte
		if masked {
			if _, err := io.ReadFull(c.r, mask[:]); err != nil {
				return nil, err
			}
		}

		payload := make([]byte, length)
		if _, err := io.ReadFull(c.r, payload); err != nil {
			return nil, err
		}
		if masked {
			for i := range payload {
				payload[i] ^= mask[i%4]
			}
		}

		switch opcode {
		case 0x8: // close
			return nil, io.EOF
		case 0x9: // ping
			c.writeFrame(0xA, payload)
			continue
		case 0xA: // pong
			continue
		case 0x1, 0x2: // text, binary
			message = payload
		case 0x0: // continuation
			message = append(message, payload...)
		default:
			return nil, fmt.Errorf("unsupported opcode %#x", opcode)
		}

		if fin && opcode != 0x9 && opcode != 0xA {
			return message, nil
		}
	}
}

func (c *wsConn) writeFrame(opcode byte, payload []byte) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	header := make([]byte, 0, 10)
	header = append(header, 0x80|opcode)

	switch {
	case len(payload) < 126:
		header = append(header, byte(len(payload)))
	case len(payload) <= 0xFFFF:
		header = append(header, 126, byte(len(payload)>>8), byte(len(payload)))
	default:
		header = append(header, 127)
		var ext [8]byte
		binary.BigEndian.PutUint64(ext[:], uint64(len(payload)))
		header = append(header, ext[:]...)
	}

	if _, err := c.conn.Write(header); err != nil {
		return err
	}
	if len(payload) > 0 {
		if _, err := c.conn.Write(payload); err != nil {
			return err
		}
	}
	return nil
}

func (c *wsConn) writeMessage(payload []byte) error {
	return c.writeFrame(0x2, payload)
}

// ---------------------------------------------------------------------------
// WISP
// ---------------------------------------------------------------------------

type stream struct {
	conn   net.Conn
	credit int
}

type session struct {
	ws      *wsConn
	mu      sync.Mutex // guards streams
	streams map[uint32]*stream
}

func (s *session) send(frameType byte, id uint32, payload []byte) {
	frame := make([]byte, 5+len(payload))
	frame[0] = frameType
	binary.LittleEndian.PutUint32(frame[1:], id)
	copy(frame[5:], payload)

	// A failed write means the peer went away, which is the normal way this
	// ends; nothing useful to report.
	_ = s.ws.writeMessage(frame)
}

func (s *session) continueStream(id uint32, credit uint32) {
	payload := make([]byte, 4)
	binary.LittleEndian.PutUint32(payload, credit)
	s.send(frameContinue, id, payload)
}

func (s *session) closeStream(id uint32, reason byte) {
	s.mu.Lock()
	st := s.streams[id]
	delete(s.streams, id)
	s.mu.Unlock()

	if st != nil && st.conn != nil {
		st.conn.Close()
	}
	s.send(frameClose, id, []byte{reason})
}

func (s *session) handleConnect(id uint32, payload []byte) {
	if len(payload) < 3 {
		s.closeStream(id, closeNetwork)
		return
	}
	streamType := payload[0]
	port := binary.LittleEndian.Uint16(payload[1:3])
	host := string(payload[3:])

	if streamType != streamTCP {
		log.Printf("stream %d: unsupported stream type %d", id, streamType)
		s.closeStream(id, closeNetwork)
		return
	}

	address := net.JoinHostPort(host, fmt.Sprint(port))
	conn, err := net.DialTimeout("tcp", address, dialTimeout)
	if err != nil {
		log.Printf("stream %d: dial %s: %v", id, address, err)
		s.closeStream(id, closeNetwork)
		return
	}

	s.mu.Lock()
	s.streams[id] = &stream{conn: conn, credit: creditWindow}
	s.mu.Unlock()

	log.Printf("stream %d: %s", id, address)

	go func() {
		buf := make([]byte, 32*1024)
		for {
			n, err := conn.Read(buf)
			if n > 0 {
				s.send(frameData, id, buf[:n])
			}
			if err != nil {
				s.closeStream(id, closeVoluntary)
				return
			}
		}
	}()
}

func (s *session) handleData(id uint32, payload []byte) {
	s.mu.Lock()
	st := s.streams[id]
	if st != nil {
		st.credit--
		spent := st.credit <= 0
		if spent {
			st.credit = creditWindow
		}
		s.mu.Unlock()

		if _, err := st.conn.Write(payload); err != nil {
			s.closeStream(id, closeNetwork)
			return
		}
		if spent {
			s.continueStream(id, creditWindow)
		}
		return
	}
	s.mu.Unlock()
}

func (s *session) run() {
	defer func() {
		s.mu.Lock()
		for id, st := range s.streams {
			st.conn.Close()
			delete(s.streams, id)
		}
		s.mu.Unlock()
		s.ws.conn.Close()
	}()

	// Without this the client never sends anything: every stream starts with
	// zero credit, so its first CONNECT sits in the client's buffer.
	s.continueStream(0, creditWindow)

	for {
		message, err := s.ws.readMessage()
		if err != nil {
			return
		}
		if len(message) < 5 {
			continue
		}

		frameType := message[0]
		id := binary.LittleEndian.Uint32(message[1:5])
		payload := message[5:]

		switch frameType {
		case frameConnect:
			s.handleConnect(id, payload)
		case frameData:
			s.handleData(id, payload)
		case frameClose:
			s.closeStream(id, closeVoluntary)
		}
	}
}

func main() {
	listen := flag.String("listen", "127.0.0.1:8080", "address to listen on")
	flag.Parse()

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ws, err := upgrade(w, r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		(&session{ws: ws, streams: map[uint32]*stream{}}).run()
	})

	server := &http.Server{Addr: *listen, Handler: handler}
	log.Printf("wisp relay on ws://%s  (point v64 at wisp://%s/)", *listen, *listen)
	if err := server.ListenAndServe(); err != nil {
		log.Fatal(err)
	}
}
